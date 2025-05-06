#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "convert_tree_md.h"
#include "yaml_parser.h"
#include "tree.h"
#include "utils.h"
#include "write_html.h"

int main(int argc, char *argv[]) {
  Node *converted_tree;
  char *source;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <markdown-path>\n", argv[0]);
    return -1;
  }

  source = read_file(argv[1]);
  if (source == NULL) {
    fprintf(stderr, "Can't open %s\n", argv[1]);
    return -1;
  }

  converted_tree = convert_tree_md(source);
  // print_tree(converted_tree);
  write_html(stdout, converted_tree);
  free_tree(converted_tree);
  free(source);
  return 0;
}
