/*
 * Write the head section based on the YAML nodes.
 *
 * See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name
 * for examples of meta tags.
 */
#include <assert.h>
#include <stdio.h>

#include "hash.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "writers/head.h"

static void _meta(FILE *file, char *name, char *content);

static void _commons_meta(FILE *file, Node *node) {
  Node *author, *description, *illustration, *title;

  author = get_key(node, "author");
  description = get_key(node, "description");
  illustration = get_key(node, "illustration");
  title = get_key(node, "title");

  fprintf(file, "<meta charset=\"utf-8\">\n");

  if (title != NULL)
    fprintf(file, "<title>%s</title>\n", title->children[0]->content);

  if (author != NULL)
    _meta(file, "author", author->children[0]->content);
  if (description != NULL)
    _meta(file, "description", description->children[0]->content);

  if (title != NULL)
    _meta(file, "og:title", title->children[0]->content);
  if (description != NULL)
    _meta(file, "og:description", description->children[0]->content);
  if (illustration != NULL)
    _meta(file, "og:image", illustration->children[0]->content);

  if (title != NULL)
    _meta(file, "twitter:title", title->children[0]->content);
  if (description != NULL)
    _meta(file, "twitter:description", description->children[0]->content);
  if (illustration != NULL)
    _meta(file, "twitter:image", illustration->children[0]->content);
}

static void _css(FILE *file, Node *tree) {
  Node *styles, *href;

  styles = get_key(tree, "styles");

  if (styles == NULL)
    return;

  for (int i = 0; i < styles->child_count; i++) {
    assert(styles->children[i]->code == HASH_BLOCK_SEQUENCE_ITEM);
    assert((href = styles->children[i]->children[0]) != NULL);
    fprintf(file, "<link rel=\"stylesheet\" href=\"%s\">\n", href->content);
  }
}

static void _scripts(FILE *file, Node *tree) {
  Node *scripts, *src;

  scripts = get_key(tree, "scripts");

  if (scripts == NULL)
    return;

  for (int i = 0; i < scripts->child_count; i++) {
    assert(scripts->children[i]->code == HASH_BLOCK_SEQUENCE_ITEM);
    assert((src = scripts->children[i]->children[0]) != NULL);
    fprintf(file, "<script src=\"%s\"></script>\n", src->content);
  }
}

static void _meta(FILE *file, char *name, char *content) {
  fprintf(file, "<meta name=\"%s\" content=\"%s\">\n", name, content);
}

void write_head(FILE *file, Node *tree) {
  fprintf(file, "<head>\n");
  _commons_meta(file, tree);
  _css(file, tree);
  _scripts(file, tree);
  fprintf(file, "</head>\n");
}
