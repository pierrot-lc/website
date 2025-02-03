#include "convert.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

const TSLanguage *tree_sitter_markdown(void);

/* Compare the parsed tree of a reference HTML file and the one produced by the
 * writer based on a typst input.
 */
int main(int argc, char *argv[]) {
  char *code_1, *code_2;
  FILE *file;

  TSParser *parser;
  TSTree *tree;
  TSNode root_node;

  if (argc < 4) {
    printf("Usage: %s <typst_path> <html_reference_path> <html_output_path>\n",
           argv[0]);
    return -1;
  }

  code_1 = read_file(argv[1]);
  assert(code_1 != NULL);

  file = fopen(argv[3], "w");
  assert(file != NULL);

  parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_markdown());
  tree = ts_parser_parse_string(parser, NULL, code_1, strlen(code_1));
  root_node = ts_tree_root_node(tree);
  convert_tree(file, code_1, root_node);

  fclose(file);
  ts_tree_delete(tree);
  free(code_1);

  code_1 = read_file(argv[2]);
  code_2 = read_file(argv[3]);
  assert(code_1 != NULL);
  assert(code_2 != NULL);
  assert(strcmp(code_1, code_2) == 0);

  ts_parser_delete(parser);
  free(code_1);
  free(code_2);
  return 0;
}
