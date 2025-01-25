import app/web.{type Context}
import gleam/string_tree
import wisp.{type Request, type Response}

pub fn handle_request(req: Request, ctx: Context) -> Response {
  use _req <- web.middleware(req, ctx)
  let body = string_tree.from_string("<h1>Hello, Joe!</h1>")

  wisp.html_response(body, 200)
}
