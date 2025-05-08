#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/markdown.h"
#include "tree.h"
#include "ts_utils.h"
#include "writers/html.h"

int main(int argc, char *argv[]) {
  Node *node;
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

  node = parse_markdown(source);
  write_html(stdout, node);
  free_tree(node);
  free(source);
  return 0;
}
