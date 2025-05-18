#include <stdio.h>

#include "parsers/yaml.h"
#include "writers/article_metadata.h"

void write_article_metadata(FILE *file, Node *tree) {
  Node *infos, *node;

  infos = get_key(tree, "infos");

  if (infos == NULL)
    return;

  for (int i = 0; i < infos->child_count; i++) {
    node = infos->children[i];
    fprintf(file, "<p class=\"article-infos\">%s: %s</p>\n", node->content,
            node->children[0]->content);
  }
}
