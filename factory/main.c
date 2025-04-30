#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "convert_tree_md.h"
#include "parse.h"
#include "tree.h"
#include "utils.h"
#include "write_html.h"

int main(int argc, char *argv[]) {
  char *source, *string;

  TSTree *tree;
  TSNode root_node;
  Node *converted_tree;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <markdown-path>\n", argv[0]);
    return -1;
  }

  source = read_file(argv[1]);
  if (source == NULL) {
    fprintf(stderr, "Can't open %s\n", argv[1]);
    return -1;
  }

  tree = parse(source, tree_sitter_markdown());
  root_node = ts_tree_root_node(tree);

  string = ts_node_string(root_node);
  printf("Syntax tree: %s\n\n", string);
  fprintf(stderr, "OK");

  converted_tree = convert_tree_md(source, tree);
  print_tree(converted_tree);
  fprintf(stderr, "OK 2");

  free(string);
  free(source);
  ts_tree_delete(tree);

  write_html(stdout, converted_tree);
  fprintf(stderr, "OK 3");
  free_tree(converted_tree);
  return 0;
}
