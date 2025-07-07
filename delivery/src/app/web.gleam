import gleam/http/request
import gleam/string
import gleam/string_tree
import wisp.{type Request, type Response}

/// Replace some default routes with their corresponding static paths.
pub fn handle_default_static(
  req: Request,
  next handler: fn(Request) -> Response,
) -> Response {
  let req = case req.path {
    "/" -> request.set_path(req, "/index.html")
    "/journal" -> request.set_path(req, "/journal.html")
    "/papers" -> request.set_path(req, "/papers.html")
    "/posts" -> request.set_path(req, "/combined.html")
    "/reviews" -> request.set_path(req, "/reviews.html")
    _ -> req
  }

  let req = case string.ends_with(req.path, "/") {
    True -> req.path |> string.append("/post.html") |> request.set_path(req, _)
    False -> req
  }

  handler(req)
}

pub fn handle_request(req: Request, static_dir: String) -> Response {
  let req = wisp.method_override(req)
  use <- wisp.log_request(req)
  use <- wisp.rescue_crashes
  use req <- wisp.handle_head(req)
  use req <- handle_default_static(req)
  use <- wisp.serve_static(req, under: "", from: static_dir)

  let body = string_tree.from_string("<h1>Hello, Joe!</h1>")
  wisp.html_response(body, 200)
}
