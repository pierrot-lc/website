#include <assert.h>
#include <stdio.h>

#include "hash.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "writers/head.h"

/**
 * Write a meta element.
 *
 * See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta/name.
 */
static void meta(FILE *file, char *name, char *content) {
  fprintf(file, "<meta name=\"%s\" content=\"%s\">\n", name, content);
}

static void commons_meta(FILE *file, Node *node) {
  Node *author, *date, *description, *illustration, *title, *favicon;

  author = get_key(node, "author");
  date = get_key(node, "date");
  description = get_key(node, "description");
  illustration = get_key(node, "illustration");
  title = get_key(node, "title");
  favicon = get_key(node, "favicon");

  fprintf(file, "<meta charset=\"utf-8\">\n");

  if (title != NULL)
    fprintf(file, "<title>%s</title>\n", get_value_scalar(title));

  if (author != NULL)
    meta(file, "author", get_value_scalar(author));
  if (description != NULL)
    meta(file, "description", get_value_scalar(description));

  if (title != NULL)
    meta(file, "og:title", get_value_scalar(title));
  if (description != NULL)
    meta(file, "og:description", get_value_scalar(description));
  if (illustration != NULL)
    meta(file, "og:image", get_value_scalar(illustration));

  if (title != NULL)
    meta(file, "twitter:title", get_value_scalar(title));
  if (description != NULL)
    meta(file, "twitter:description", get_value_scalar(description));
  if (illustration != NULL)
    meta(file, "twitter:image", get_value_scalar(illustration));

  if (date != NULL)
    meta(file, "published-date", get_value_scalar(date));

  if (favicon != NULL)
    fprintf(file, "<link rel=\"icon\" href=\"%s\" type=\"image/x-icon\">\n",
            get_value_scalar(favicon));
}

static void css(FILE *file, Node *tree) {
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

static void scripts(FILE *file, Node *tree) {
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

void write_head(FILE *file, Node *tree) {
  fprintf(file, "<head>\n");
  commons_meta(file, tree);
  css(file, tree);
  scripts(file, tree);
  fprintf(file, "</head>\n");
}
