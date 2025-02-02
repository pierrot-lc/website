#include "writer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

const TSLanguage *tree_sitter_typst(void);

char *read_file(const char *path) {
  FILE *file;
  char *content;
  long length;

  if ((file = fopen(path, "r")) == NULL) {
    printf("Can't open %s", path);
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  length = ftell(file);
  fseek(file, 0, SEEK_SET);

  content = (char *)malloc(length + 1);
  if (content == NULL) {
    fclose(file);
    return NULL;
  }

  size_t read_length = fread(content, sizeof(char), length, file);
  if (read_length != length) {
    free(content);
    fclose(file);
    return NULL;
  }

  content[length] = '\0';

  fclose(file);
  return content;
}

int main(int argc, char *argv[]) {
  char *typst_code, *expected_html_code, *produced_html_code;
  FILE *file;

  TSParser *parser;
  TSTree *tree;
  TSNode root_node;

  if (argc < 4) {
    printf("Usage: %s <typst_path> <expected_html_path> <produced_html_path>\n",
           argv[0]);
    return -1;
  }

  typst_code = read_file(argv[1]);
  assert(typst_code != NULL);

  expected_html_code = read_file(argv[2]);
  assert(expected_html_code != NULL);

  file = fopen(argv[3], "w");
  assert(file != NULL);

  parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_typst());
  tree = ts_parser_parse_string(parser, NULL, typst_code, strlen(typst_code));
  root_node = ts_tree_root_node(tree);

  write_tree(file, typst_code, root_node);
  fclose(file);

  produced_html_code = read_file(argv[3]);
  assert(produced_html_code != NULL);

  assert(strcmp(produced_html_code, expected_html_code) == 0);

  ts_tree_delete(tree);
  ts_parser_delete(parser);
  free(produced_html_code);
  free(expected_html_code);
  free(typst_code);
  return 0;
}
