#include "hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

#define HASH_CONTENT 229462175750528
#define HASH_HEADING 229468225748821
#define HASH_SECTION 229482434843354
#define HASH_SOURCE_FILE 13894110916739591349
#define HASH_TEXT 6385723658

const TSLanguage *tree_sitter_typst(void);

void convert_tree(TSNode root_node) {
  const unsigned long root_hash = hash(ts_node_type(root_node));

  switch (root_hash) {
  case HASH_HEADING:
    printf("Hey!\n");
    break;

  case HASH_SECTION:
    printf("Section baby\n");
    break;

  default:
    printf("%s\n", ts_node_type(root_node));
    break;
  }

  for (int i = 0; i < ts_node_child_count(root_node); i++)
    convert_tree(ts_node_child(root_node, i));
}

int main() {
  TSParser *parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_typst());

  const char *source_code = "= Title\nssdq";
  TSTree *tree =
      ts_parser_parse_string(parser, NULL, source_code, strlen(source_code));

  TSNode root_node = ts_tree_root_node(tree);

  char *string = ts_node_string(root_node);
  printf("Syntax tree: %s\n", string);

  convert_tree(root_node);

  free(string);
  ts_tree_delete(tree);
  ts_parser_delete(parser);
  return 0;
}
