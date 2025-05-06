#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"
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

  // converted_tree = convert_tree_md(source);
  converted_tree = parse_yaml(source);
  print_tree(converted_tree);
  printf("Value of key '%s': %s\n", "title", get_value(converted_tree, "title")->content);
  // write_html(stdout, converted_tree);
  free_tree(converted_tree);
  free(source);
  return 0;
}
