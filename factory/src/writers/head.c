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
  Node *title = get_value(tree_root(node), "title");

  if (title == NULL)
    return;

  fprintf(file, "<title>%s</title>\n", title->content);
  _meta(file, "og:title", title->content);
  _meta(file, "twitter:title", title->content);
}

static void _meta(FILE *file, char *name, char *content) {
  fprintf(file, "<meta name=\"%s\" content=\"%s\">\n", name, content);
}

void write_head(FILE *file, Node *node) {
  fprintf(file, "<head>\n");
  _charset(file, node);
  _title(file, node);
  fprintf(file, "</head>\n");
}
