import gleam/http/request
import gleam/string_tree
import wisp.{type Request, type Response}

/// Replace some default routes with their corresponding static paths.
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

pub fn handle_request(req: Request) -> Response {
  let req = wisp.method_override(req)
  let assert Ok(priv_directory) = wisp.priv_directory("delivery")

  use <- wisp.log_request(req)
  use <- wisp.rescue_crashes
  use req <- wisp.handle_head(req)
  use req <- handle_default_static(req)
  use <- wisp.serve_static(req, under: "", from: priv_directory)

  let body = string_tree.from_string("<h1>Hello, Joe!</h1>")
  wisp.html_response(body, 200)
}
