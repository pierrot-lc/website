#include "convert.h"
#include "hash.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <tree_sitter/api.h>

void convert_tree(FILE *file, const char *source, TSNode node) {
  switch (hash(ts_node_type(node))) {
  case HASH_ATX_HEADING:
    convert_heading(file, source, node);
    break;

  case HASH_INLINE:
    convert_inline(file, source, node);
    break;

  case HASH_PARAGRAPH:
    fprintf(file, "<p>");
    convert_named_children(file, source, node);
    fprintf(file, "</p>\n");
    break;

  default:
    convert_named_children(file, source, node);
  }
}

void convert_heading(FILE *file, const char *source, TSNode node) {
  assert(hash(ts_node_type(node)) == HASH_ATX_HEADING);
  assert(ts_node_child_count(node) == 2);

  TSNode marker = ts_node_child(node, 0);
  TSNode content = ts_node_child(node, 1);

  switch ((hash(ts_node_type(marker)))) {
  case HASH_ATX_H1_MARKER:
    fprintf(file, "<h1>");
    convert_inline(file, source, content);
    fprintf(file, "</h1>\n");
    break;

  default:
    assert(false);
  }
}

void convert_inline(FILE *file, const char *source, TSNode node) {
  assert(hash(ts_node_type(node)) == HASH_INLINE);

  char *text = node_text(source, node);
  fprintf(file, "%s", text);
  free(text);
}

void convert_named_children(FILE *file, const char *source, TSNode node) {
  for (int i = 0; i < ts_node_named_child_count(node); i++)
    convert_tree(file, source, ts_node_named_child(node, i));
}
