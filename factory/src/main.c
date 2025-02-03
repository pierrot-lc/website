#include "convert.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

const TSLanguage *tree_sitter_markdown(void);

int main(int argc, char *argv[]) {
  char *source, *string;

  TSParser *parser = ts_parser_new();
  TSTree *tree;
  TSNode root_node;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <markdown-path>\n", argv[0]);
    return -1;
  }

  source = read_file(argv[1]);
  if (source == NULL) {
    fprintf(stderr, "Can't open %s\n", argv[1]);
    return -1;
  }

  ts_parser_set_language(parser, tree_sitter_markdown());
  tree = ts_parser_parse_string(parser, NULL, source, strlen(source));
  root_node = ts_tree_root_node(tree);

  string = ts_node_string(root_node);
  printf("Syntax tree: %s\n\n", string);
  convert_tree(stdout, source, root_node);

  free(string);
  free(source);
  ts_tree_delete(tree);
  ts_parser_delete(parser);
  return 0;
}
