import app/context.{type Context}
import gleam/http/request
import gleam/string_tree
import wisp.{type Request, type Response}

pub fn handle_request(req: Request, ctx: Context) -> Response {
  use _req <- middleware(req, ctx)
  let body = string_tree.from_string("<h1>Hello, Joe!</h1>")

  wisp.html_response(body, 200)
}

pub fn handle_default_static(
  req: Request,
  next handler: fn(Request) -> Response,
) -> Response {
  let req = case req.path {
    "/" -> request.set_path(req, "/index.html")
    _ -> req
  }

  handler(req)
}

/// The middleware stack that the request handler uses. The stack is itself a
/// middleware function!
///
/// Middleware wrap each other, so the request travels through the stack from
/// top to bottom until it reaches the request handler, at which point the
/// response travels back up through the stack.
///
/// The middleware used here are the ones that are suitable for use in your
/// typical web application.
///
pub fn middleware(
  req: Request,
  ctx: Context,
  handle_request: fn(Request) -> Response,
) -> Response {
  let req = wisp.method_override(req)
  use <- wisp.log_request(req)
  use <- wisp.rescue_crashes
  use req <- wisp.handle_head(req)
  use req <- handle_default_static(req)
  use <- wisp.serve_static(req, under: "", from: "priv")

  handle_request(req)
}
