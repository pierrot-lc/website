#include "convert.h"
#include "utils.h"
#include <assert.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

const TSLanguage *tree_sitter_markdown(void);

/* Compare a reference HTML file and one converted from a corresponding
 * markdown reference.
 */
int main(int argc, char *argv[]) {
  char *code_1, *code_2;
  char path[100];
  FILE *file;

  TSParser *parser;
  TSTree *tree;
  TSNode root_node;

  if (argc < 3) {
    printf("Usage: %s MD_FILE HTML_FILE\n", argv[0]);
    return -1;
  }

  code_1 = read_file(argv[1]);
  assert(code_1 != NULL);

  sprintf(path, "/tmp/%s", basename(argv[2]));
  file = fopen(path, "w");
  assert(file != NULL);

  parser = ts_parser_new();
  assert(ts_parser_set_language(parser, tree_sitter_markdown()));
  tree = ts_parser_parse_string(parser, NULL, code_1, strlen(code_1));
  root_node = ts_tree_root_node(tree);
  convert_tree(file, code_1, root_node);

  fclose(file);
  free(code_1);
  ts_tree_delete(tree);
  ts_parser_delete(parser);

  code_1 = read_file(argv[2]);
  code_2 = read_file(path);
  assert(code_1 != NULL);
  assert(code_2 != NULL);

  if (strcmp(code_1, code_2) != 0)
    fprintf(stderr, "%s\n%s\n", code_1, code_2);
  assert(strcmp(code_1, code_2) == 0);

  free(code_2);
  free(code_1);
  remove(path);
  return 0;
}
