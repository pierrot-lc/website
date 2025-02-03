#include "convert.h"
#include "hash.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

const TSLanguage *tree_sitter_markdown_inline(void);

void convert_tree(FILE *file, const char *source, TSNode node) {
  switch (hash(ts_node_type(node))) {
  case HASH_ATX_HEADING:
    _convert_heading(file, source, node);
    break;

  case HASH_INLINE:
    convert_inline_tree(file, source, node);
    break;

  case HASH_PARAGRAPH:
    fprintf(file, "<p>");
    _convert_named_children(file, source, node);
    fprintf(file, "</p>\n");
    break;

  default:
    _convert_named_children(file, source, node);
  }
}

void convert_inline_tree(FILE *file, const char *source, TSNode node) {
  assert(hash(ts_node_type(node)) == HASH_INLINE);
  assert(ts_node_named_child_count(node) == 0);

  TSParser *parser = ts_parser_new();
  TSTree *tree;
  char *text = node_text(source, node);

  ts_parser_set_language(parser, tree_sitter_markdown_inline());
  tree = ts_parser_parse_string(parser, NULL, text, strlen(text));
  _convert_inline(file, text, ts_tree_root_node(tree));

  free(text);
  ts_tree_delete(tree);
  ts_parser_delete(parser);
}

void _convert_emphasis(FILE *file, const char *source, TSNode node) {
  assert(ts_node_named_child_count(node) == 2);

  TSNode node_1 = ts_node_named_child(node, 0);
  TSNode node_2 = ts_node_named_child(node, 1);
  char *emph_1 = node_text(source, node_1);
  char *emph_2 = node_text(source, node_2);
  char *text = extract_text(source, ts_node_end_byte(node_1),
                            ts_node_start_byte(node_2));
  assert(strcmp(emph_1, emph_2) == 0);

  if (strcmp(emph_1, "*") == 0) {
    fprintf(file, "<strong>%s</strong>", text);
  } else if (strcmp(emph_2, "_") == 0) {
    fprintf(file, "<em>%s</em>", text);
  } else
    assert(false);

  free(text);
  free(emph_2);
  free(emph_1);
}

void _convert_heading(FILE *file, const char *source, TSNode node) {
  assert(hash(ts_node_type(node)) == HASH_ATX_HEADING);
  assert(ts_node_named_child_count(node) == 2);

  TSNode marker = ts_node_child(node, 0);
  TSNode content = ts_node_child(node, 1);

  switch ((hash(ts_node_type(marker)))) {
  case HASH_ATX_H1_MARKER:
    fprintf(file, "<h1>");
    convert_inline_tree(file, source, content);
    fprintf(file, "</h1>\n");
    break;

  default:
    assert(false);
  }
}

void _convert_inline(FILE *file, const char *source, TSNode node) {
  char *text;
  TSNode child;
  unsigned int start = ts_node_start_byte(node);
  unsigned int end;

  for (int i = 0; i < ts_node_named_child_count(node); i++) {
    child = ts_node_named_child(node, i);

    // Copy the content from the current start to the next child.
    end = ts_node_start_byte(child);
    text = extract_text(source, start, end);
    fprintf(file, "%s", text);
    free(text);

    // Convert the child.
    switch (hash(ts_node_type(child))) {
    case HASH_EMPHASIS:
      _convert_emphasis(file, source, child);
      break;

    default:
      fprintf(stderr, "%s", ts_node_type(child));
      assert(false);
    }

    // Update starting point for next iteration.
    start = ts_node_end_byte(child);
  }

  // Copy the rest of the inline node.
  end = ts_node_end_byte(node);
  text = extract_text(source, start, end);
  fprintf(file, "%s", text);
  free(text);
}

void _convert_named_children(FILE *file, const char *source, TSNode node) {
  for (int i = 0; i < ts_node_named_child_count(node); i++)
    convert_tree(file, source, ts_node_named_child(node, i));
}
