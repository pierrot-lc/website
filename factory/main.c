#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/markdown.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"
#include "writers/html.h"

int main(int argc, char *argv[]) {
  char *source;

  Node *node;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <markdown-path>\n", argv[0]);
    return -1;
  }

  source = read_file("/home/pierrot-lc/GitHub/website/website/global.yaml");
  node = parse_yaml(source);
  articles_index_page(stdout, "../website/articles", node);
  return 0;

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
