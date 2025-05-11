/*
 * Write the head section based on the YAML nodes.
 *
 * See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name
 * for examples of meta tags.
 */
#include <stdio.h>

#include "parsers/yaml.h"
#include "tree.h"
#include "writers/head.h"

static void _meta(FILE *file, char *name, char *content);

static void _charset(FILE *file, Node *node) {
  fprintf(file, "<meta charset=\"utf-8\">\n");
}

static void _title(FILE *file, Node *node) {
  Node *title = get_key(tree_root(node), "title");

  if (title == NULL)
    return;

  fprintf(file, "<title>%s</title>\n", title->children[0]->content);
  _meta(file, "og:title", title->children[0]->content);
  _meta(file, "twitter:title", title->children[0]->content);
}

static void _css(FILE *file, Node *tree) {
  Node *styles;

  styles = get_key(tree, "styles");

  if (styles == NULL)
    return;

  for (int i = 0; i < styles->child_count; i++)
    fprintf(file, "<link rel=\"stylesheet\" href=\"%s\">\n",
            styles->children[i]->content);
}

static void _meta(FILE *file, char *name, char *content) {
  fprintf(file, "<meta name=\"%s\" content=\"%s\">\n", name, content);
}

void write_head(FILE *file, Node *tree) {
  fprintf(file, "<head>\n");
  _charset(file, tree);
  _title(file, tree);
  _css(file, tree);
  fprintf(file, "</head>\n");
}
