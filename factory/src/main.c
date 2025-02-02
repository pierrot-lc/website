#include "writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

const TSLanguage *tree_sitter_typst(void);

int main() {
  TSParser *parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_typst());

  const char *source_code = "== Hey!!\nssdq";
  TSTree *tree =
      ts_parser_parse_string(parser, NULL, source_code, strlen(source_code));

  TSNode root_node = ts_tree_root_node(tree);

  char *string = ts_node_string(root_node);
  printf("Syntax tree: %s\n", string);

  FILE *file = fopen("test.html", "w");
  if (file == NULL) {
    printf("Error opening the file");
    return -1;
  }

  write_tree(file, source_code, root_node);

  fclose(file);
  free(string);
  ts_tree_delete(tree);
  ts_parser_delete(parser);
  return 0;
}
