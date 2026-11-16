import { createHash } from "node:crypto";

import { getOrgContext } from "@agent-native/core/org";
import {
  getAppProductionUrl,
  getBrowserCookieSession,
  getForwardedRequestOrigin,
  getSession,
  getRequestIdentityAuthenticatedAtMs,
  isSessionResolutionUnavailable,
  resolveAppRuntimeUrl,
  runWithRequestContext,
} from "@agent-native/core/server";
import {
  EMBED_START_PATH,
  EMBED_TOKEN_QUERY_PARAM,
} from "@agent-native/core/shared";

import { parseWorkspaceAppChatProxyPath } from "../../shared/workspace-app-chat.js";
import { listWorkspaceApps } from "./app-creation-store.js";
import {
  createGrantedDispatchMcpEmbedSessionForWorkspaceAppChat,
  createWorkspaceSsoEmbedSessionForWorkspaceAppChat,
  mountedWorkspaceAppHomeUrl,
} from "./mcp-gateway.js";

const TARGET_CHAT_PATH = "/_agent-native/agent-chat";
/**
 * Browser-supplied document target selectors never cross the chat proxy.
 */
const EMBED_TARGET_HEADER = "x-agent-native-embed-target";
const CSRF_MARKER_HEADER = "x-agent-native-csrf";
/** Re-mint this far before the token's own expiry so a long turn cannot age out mid-stream. */
const SESSION_REFRESH_MARGIN_MS = 5 * 60 * 1000;

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

const DROPPED_REQUEST_HEADERS = new Set([
  "accept-encoding",
  "authorization",
  "content-length",
  "cookie",
  "host",
  "origin",
  "referer",
  EMBED_TARGET_HEADER,
]);

const DROPPED_RESPONSE_HEADERS = new Set([
  "content-encoding",
  "content-length",
  "set-cookie",
  "transfer-encoding",
]);

export interface WorkspaceAppChatSession {
  appBaseUrl: string;
  token: string;
  expiresAt: number;
}

const SESSION_CACHE_MAX_ENTRIES = 128;
// Bound in-flight retention as well as completed grants; minting already has a 95s budget.
const SESSION_CACHE_PENDING_TTL_MS = 100_000;
interface CachedChatSession {
  pending: Promise<WorkspaceAppChatSession>;
  expiresAt: number;
}
const sessionCache = new Map<string, CachedChatSession>();
/**
 * Concurrent polls of mode/runs/list can share one registry read. The
 * promise is dropped as soon as it settles, so a later request always
 * re-checks access instead of reusing a grant.
 */
const browserAppUrlInFlight = new Map<string, Promise<string | null>>();

function sweepChatSessionCache(now: number): void {
  for (const [key, entry] of sessionCache) {
    if (entry.expiresAt <= now) sessionCache.delete(key);
  }
}

interface WorkspaceAppChatProxyEvent {
  url: URL;
  req: Request;
  context?: Record<string, unknown>;
}

function sessionCacheKey(input: {
  appId: string;
  userEmail: string;
  orgId: string | undefined;
  requestOrigin: string;
  identitySessionKey: string;
}): string {
  return [
    input.userEmail,
    input.orgId ?? "",
    input.requestOrigin,
    input.appId,
    input.identitySessionKey,
  ].join("\u0000");
}

type HttpOriginResult =
  | { kind: "absent" }
  | { kind: "invalid"; cause: unknown }
  | { kind: "valid"; origin: string };

function httpOrigin(value: string | undefined): HttpOriginResult {
  if (value === undefined) return { kind: "absent" };
  try {
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return {
        kind: "invalid",
        cause: new Error("Application origin must use HTTP or HTTPS."),
      };
    }
    return { kind: "valid", origin: url.origin };
  } catch (cause) {
    return { kind: "invalid", cause };
  }
}

function resolveTrustedWorkspaceRequestOrigin(
  event: WorkspaceAppChatProxyEvent,
): { origin: string; requestHostTrusted: boolean } {
  const configured = httpOrigin(getAppProductionUrl());
  const runtime = httpOrigin(resolveAppRuntimeUrl());
  const request = httpOrigin(
    getForwardedRequestOrigin(
      event as unknown as Parameters<typeof getForwardedRequestOrigin>[0],
    ),
  );
  const allowedOrigins = new Set(
    [configured, runtime]
      .filter(
        (result): result is Extract<HttpOriginResult, { kind: "valid" }> =>
          result.kind === "valid",
      )
      .map((result) => result.origin),
  );

  if (request.kind === "valid" && allowedOrigins.has(request.origin)) {
    return { origin: request.origin, requestHostTrusted: true };
  }
  if (configured.kind === "valid") {
    return { origin: configured.origin, requestHostTrusted: false };
  }
  if (runtime.kind === "valid") {
    return { origin: runtime.origin, requestHostTrusted: false };
  }
  if (configured.kind === "invalid" || runtime.kind === "invalid") {
    throw new Error("Configured application origin is invalid.");
  }
  throw new Error("No trusted application origin is configured for app chat.");
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : typeof error === "string" || typeof error === "number"
      ? String(error)
      : JSON.stringify(error);
}

/**
 * Read the embed token's own `exp` claim. Guessing a TTL here would silently
 * produce a session that the target has already rejected, so a token we cannot
 * read an expiry from is an error rather than a default.
 */
export function embedSessionTokenExpiry(token: string): number {
  const payload = token.split(".")[0];
  if (!payload) throw new Error("Embed session token is malformed.");
  let claims: unknown;
  try {
    claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    throw new Error("Embed session token payload could not be decoded.");
  }
  const exp = (claims as { exp?: unknown } | null)?.exp;
  if (typeof exp !== "number" || !Number.isFinite(exp)) {
    throw new Error("Embed session token has no usable expiry.");
  }
  return exp * 1000;
}

async function mintChatSession(
  appId: string,
): Promise<WorkspaceAppChatSession> {
  try {
    return await createWorkspaceSsoEmbedSessionForWorkspaceAppChat(appId);
  } catch (ssoError) {
    if (
      ssoError &&
      typeof ssoError === "object" &&
      (("name" in ssoError &&
        ssoError.name === "TargetMcpAuthenticationRejectedError") ||
        ("status" in ssoError &&
          (ssoError.status === 401 || ssoError.status === 403)) ||
        ("statusCode" in ssoError &&
          (ssoError.statusCode === 401 || ssoError.statusCode === 403)))
    )
      throw ssoError;
    try {
      return await createGrantedDispatchMcpEmbedSessionForWorkspaceAppChat(
        appId,
      );
    } catch (grantError) {
      throw new Error(
        `Could not create an app session for "${appId}". ` +
          `Workspace sign-in: ${errorMessage(ssoError)}. ` +
          `Dispatch app grant: ${errorMessage(grantError)}.`,
      );
    }
  }
}

export async function exchangeEmbedStartUrl(
  startUrl: string,
  expectedAppBaseUrl: string,
  fetchImpl: typeof fetch = fetch,
): Promise<WorkspaceAppChatSession> {
  const start = new URL(startUrl);
  const startIndex = start.pathname.lastIndexOf(EMBED_START_PATH);
  if (startIndex < 0) {
    throw new Error("App session start URL is not an embed start route.");
  }
  const response = await fetchImpl(start.toString(), { redirect: "manual" });
  await response.body?.cancel().catch(() => undefined);
  const location = response.headers.get("location");
  if (response.status < 300 || response.status >= 400 || !location) {
    throw new Error(
      `App session exchange failed with status ${response.status}.`,
    );
  }

  const resolved = new URL(location, start);
  if (resolved.origin !== start.origin) {
    throw new Error("App session exchange redirected to an untrusted origin.");
  }
  const expectedApp = new URL(expectedAppBaseUrl);
  if (
    expectedApp.origin !== start.origin ||
    !isWithinAppMount(expectedApp.pathname, resolved.pathname)
  ) {
    throw new Error("App session exchange redirected outside its app mount.");
  }
  const token = resolved.searchParams.get(EMBED_TOKEN_QUERY_PARAM);
  if (!token) {
    throw new Error("App session exchange returned no embed token.");
  }
  return {
    appBaseUrl: `${expectedApp.origin}${expectedApp.pathname.replace(/\/+$/, "")}`,
    token,
    expiresAt: embedSessionTokenExpiry(token),
  };
}

function appPathSegments(pathname: string): string[] | null {
  return pathname.split("/").filter(Boolean);
}

function isWithinAppMount(mountPath: string, targetPath: string): boolean {
  const mount = appPathSegments(mountPath);
  const target = appPathSegments(targetPath);
  return (
    mount !== null &&
    target !== null &&
    mount.length <= target.length &&
    mount.every((segment, index) => target[index] === segment)
  );
}

async function resolveWorkspaceAppChatSession(input: {
  identityAuthenticatedAtMs: number;
  identitySessionKey: string | undefined;
  appId: string;
  userEmail: string;
  orgId: string | undefined;
  requestOrigin: string;
  fetchImpl: typeof fetch;
}): Promise<WorkspaceAppChatSession> {
  sweepChatSessionCache(Date.now());
  // Authentication timestamps can be request-scoped, so only a verified stable
  // session token may key a retained grant. Token-less identities mint per request.
  const key = input.identitySessionKey
    ? sessionCacheKey({
        ...input,
        identitySessionKey: input.identitySessionKey,
      })
    : undefined;
  const cached = key ? sessionCache.get(key) : undefined;
  if (cached) {
    try {
      const session = await cached.pending;
      if (session.expiresAt - Date.now() > SESSION_REFRESH_MARGIN_MS) {
        return session;
      }
      // coercion-ok: a mint that already failed is retried below, and that fresh
      // attempt reports its own error rather than replaying a stale one.
    } catch {
      void 0;
    }
    if (key && sessionCache.get(key) === cached) sessionCache.delete(key);
  }

  const pending = Promise.resolve(
    runWithRequestContext(
      {
        userEmail: input.userEmail,
        identityAuthenticatedAtMs: input.identityAuthenticatedAtMs,
        ...(input.orgId ? { orgId: input.orgId } : {}),
        requestOrigin: input.requestOrigin,
      },
      async () => {
        const session = await mintChatSession(input.appId);
        const expiresAt = embedSessionTokenExpiry(session.token);
        if (expiresAt <= Date.now())
          throw new Error("App chat credential has expired.");
        return {
          ...session,
          expiresAt: Math.min(expiresAt, session.expiresAt),
        };
      },
    ),
  );
  const entry = {
    pending,
    expiresAt: Date.now() + SESSION_CACHE_PENDING_TTL_MS,
  };
  if (key) {
    for (const oldest of sessionCache.keys()) {
      if (sessionCache.size < SESSION_CACHE_MAX_ENTRIES) break;
      sessionCache.delete(oldest);
    }
    sessionCache.set(key, entry);
  }
  try {
    const session = await pending;
    entry.expiresAt = session.expiresAt - SESSION_REFRESH_MARGIN_MS;
    return session;
  } catch (error) {
    if (key && sessionCache.get(key) === entry) sessionCache.delete(key);
    throw error;
  }
}

export function clearWorkspaceAppChatSessions(): void {
  sessionCache.clear();
  browserAppUrlInFlight.clear();
}

function browserAppUrlKey(input: {
  appId: string;
  userEmail: string;
  orgId: string | undefined;
  requestOrigin: string;
}): string {
  return [
    input.userEmail,
    input.orgId ?? "",
    input.requestOrigin,
    input.appId,
  ].join("\u0000");
}

function resolveBrowserAppUrl(input: {
  appId: string;
  userEmail: string;
  orgId: string | undefined;
  requestOrigin: string;
}): Promise<string | null> {
  const key = browserAppUrlKey(input);
  const inFlight = browserAppUrlInFlight.get(key);
  if (inFlight) return inFlight;
  const url = Promise.resolve(
    runWithRequestContext(
      {
        userEmail: input.userEmail,
        ...(input.orgId ? { orgId: input.orgId } : {}),
        requestOrigin: input.requestOrigin,
      },
      () => listWorkspaceApps({ includeAgentCards: false }),
    ),
  ).then((apps) => {
    const app = apps.find(
      (candidate) =>
        candidate.id === input.appId &&
        !candidate.isDispatch &&
        candidate.status !== "pending",
    );
    return app ? mountedWorkspaceAppHomeUrl(app) : null;
  });
  browserAppUrlInFlight.set(key, url);
  const forget = () => {
    if (browserAppUrlInFlight.get(key) === url) {
      browserAppUrlInFlight.delete(key);
    }
  };
  url.then(forget, forget);
  return url;
}

function isWorkspaceAppChatTarget(appBaseUrl: string, target: URL): boolean {
  // URL parsing resolves encoded dot segments such as "%2e%2e", so check where
  // the request lands before it carries the caller's session.
  const chatPath = `${new URL(appBaseUrl).pathname.replace(/\/+$/, "")}${TARGET_CHAT_PATH}`;
  const chatSubPath = target.pathname.slice(chatPath.length);
  return (
    target.pathname.startsWith(chatPath) &&
    (chatSubPath === "" || chatSubPath.startsWith("/")) &&
    !/%(?:2f|5c)/i.test(chatSubPath)
  );
}

function upstreamRequestHeaders(
  request: Request,
  session?: WorkspaceAppChatSession,
): Headers {
  const headers = new Headers();
  for (const [name, value] of request.headers) {
    const lower = name.toLowerCase();
    const browserHeader =
      !session &&
      (lower === "cookie" || lower === "origin" || lower === "referer");
    if (
      HOP_BY_HOP_HEADERS.has(lower) ||
      (DROPPED_REQUEST_HEADERS.has(lower) && !browserHeader)
    ) {
      continue;
    }
    headers.set(name, value);
  }
  if (session) {
    headers.set("authorization", `Bearer ${session.token}`);
    headers.set(CSRF_MARKER_HEADER, "1");
  }
  return headers;
}

function proxyErrorResponse(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

export function createWorkspaceAppChatProxyHandler(
  options: { fetchImpl?: typeof fetch } = {},
) {
  const fetchImpl = options.fetchImpl ?? fetch;
  return async (event: WorkspaceAppChatProxyEvent): Promise<Response> => {
    const url = event.url;
    const rawMountedPathname = event.context?._mountedPathname;
    const mountedPathname =
      typeof rawMountedPathname === "string" ? rawMountedPathname : undefined;
    const parsed = parseWorkspaceAppChatProxyPath(
      mountedPathname ?? url.pathname,
    );
    if (!parsed) {
      return proxyErrorResponse(404, "Workspace app chat route not found.");
    }

    let session: Awaited<ReturnType<typeof getSession>>;
    let orgId: string | undefined;
    try {
      session = await getSession(event as never);
      orgId = (await getOrgContext(event as never)).orgId ?? undefined;
    } catch (error) {
      return proxyErrorResponse(
        503,
        `Dispatch could not read your workspace session: ${errorMessage(error)}`,
      );
    }
    if (!session?.email) {
      return proxyErrorResponse(401, "Sign in to Dispatch to use app chat.");
    }

    const identityAuthenticatedAtMs = getRequestIdentityAuthenticatedAtMs(
      event,
      session.email,
    );
    if (
      typeof identityAuthenticatedAtMs !== "number" ||
      !Number.isSafeInteger(identityAuthenticatedAtMs) ||
      identityAuthenticatedAtMs <= 0 ||
      identityAuthenticatedAtMs > Date.now()
    ) {
      return proxyErrorResponse(
        401,
        "Sign in to Dispatch again to use app chat.",
      );
    }

    const identitySessionKey =
      typeof session.token === "string" && session.token
        ? `session:${createHash("sha256").update(session.token).digest("hex")}`
        : undefined;

    let requestOrigin: string;
    let appBaseUrl: string;
    let appSession: WorkspaceAppChatSession | undefined;
    try {
      const originDecision = resolveTrustedWorkspaceRequestOrigin(event);
      requestOrigin = originDecision.origin;
      const browserSession = await getBrowserCookieSession(event as never);
      if (
        browserSession === null &&
        isSessionResolutionUnavailable(event as never)
      ) {
        throw new Error("Dispatch could not read your browser session.");
      }
      let browserAppUrl: string | null = null;
      if (
        originDecision.requestHostTrusted &&
        browserSession?.email === session.email &&
        (browserSession.orgId ?? null) === (orgId ?? null)
      ) {
        browserAppUrl = await resolveBrowserAppUrl({
          appId: parsed.appId,
          userEmail: session.email,
          orgId,
          requestOrigin,
        });
      }
      // Page-only embed tokens do not authorize chat; same-origin apps
      // recheck the browser session themselves.
      if (browserAppUrl && new URL(browserAppUrl).origin === requestOrigin) {
        appBaseUrl = browserAppUrl;
      } else {
        appSession = await resolveWorkspaceAppChatSession({
          appId: parsed.appId,
          userEmail: session.email,
          identityAuthenticatedAtMs,
          identitySessionKey,
          orgId,
          requestOrigin,
          fetchImpl,
        });
        appBaseUrl = appSession.appBaseUrl;
      }
    } catch (error) {
      console.warn("[dispatch] workspace app chat proxy could not connect", {
        app: parsed.appId,
        error: errorMessage(error),
      });
      return proxyErrorResponse(
        502,
        `${parsed.appId} chat is unavailable: ${errorMessage(error)}`,
      );
    }

    const request = event.req;
    const target = new URL(
      `${appBaseUrl}${TARGET_CHAT_PATH}${parsed.targetSubPath}`,
    );
    target.search = url.search;
    if (!isWorkspaceAppChatTarget(appBaseUrl, target)) {
      return proxyErrorResponse(404, "Workspace app chat route not found.");
    }

    const method = request.method.toUpperCase();
    const body =
      method === "GET" || method === "HEAD"
        ? undefined
        : await request.arrayBuffer();

    let upstream: Response;
    try {
      upstream = await fetchImpl(target.toString(), {
        method,
        headers: upstreamRequestHeaders(request, appSession),
        ...(body === undefined ? {} : { body }),
        redirect: "manual",
      });
    } catch (error) {
      console.warn("[dispatch] workspace app chat proxy request failed", {
        app: parsed.appId,
        targetPath: target.pathname,
        error: errorMessage(error),
      });
      return proxyErrorResponse(
        502,
        `${parsed.appId} chat did not respond: ${errorMessage(error)}`,
      );
    }

    if (
      identitySessionKey &&
      (upstream.status === 401 || upstream.status === 403)
    ) {
      sessionCache.delete(
        sessionCacheKey({
          appId: parsed.appId,
          userEmail: session.email,
          orgId,
          requestOrigin,
          identitySessionKey,
        }),
      );
    }

    const headers = new Headers();
    for (const [name, value] of upstream.headers) {
      const lower = name.toLowerCase();
      if (
        HOP_BY_HOP_HEADERS.has(lower) ||
        DROPPED_RESPONSE_HEADERS.has(lower)
      ) {
        continue;
      }
      headers.set(name, value);
    }
    // `upstream.body` is the live stream: SSE run events and the turn's token
    // stream reach the browser as the app emits them, never buffered here.
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  };
}
