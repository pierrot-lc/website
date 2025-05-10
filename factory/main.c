#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/markdown.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"
#include "writers/html.h"

int from_markdown(char *markdown_path, char *config_path) {
  char *source;

  Node *yaml, *markdown;

  source = read_file(config_path);
  if (source == NULL) {
    fprintf(stderr, "Can't open %s\n", config_path);
    return -1;
  }
  yaml = parse_yaml(source);
  free(source);

  source = read_file(markdown_path);
  if (source == NULL) {
    fprintf(stderr, "Can't open %s\n", config_path);
    return -1;
  }
  markdown = parse_markdown(source);
  free(source);

  // NOTE: Merge the yaml configuration with the parsed markdown tree. Any yaml
  // configuration found in the markdown tree will overwrite the yaml
  // configuration since `get_value` is DFS.
  add_child(markdown, yaml);
  write_html(stdout, markdown);

  free_tree(markdown);

  return 0;
}

int articles_index(char *articles_dir, char *config_path) {
  char *source;

  Node *yaml;

  source = read_file(config_path);
  if (source == NULL) {
    fprintf(stderr, "Can't open %s\n", config_path);
    return -1;
  }
  yaml = parse_yaml(source);
  free(source);

  articles_index_page(stdout, articles_dir, yaml);

  return 0;
}

void help_args(char *self) {
  fprintf(stdout,
          "Usage: %s --config <config-path> [--md <md-path> ] [--articles "
          "<articles-dir>]\n",
          self);
}

int main(int argc, char *argv[]) {
  char *markdown_path = NULL, *articles_dir = NULL, *config_path = NULL;

  for (int i = 1; i < argc;) {
    if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[i + 1];
      i += 2;
      continue;
    }

    if (strcmp(argv[i], "--md") == 0 && i + 1 < argc) {
      markdown_path = argv[i + 1];
      i += 2;
      continue;
    }

    if (strcmp(argv[i], "--articles") == 0 && i + 1 < argc) {
      articles_dir = argv[i + 1];
      i += 2;
      continue;
    }

    help_args(argv[0]);
    return 0;
  }

  if (config_path == NULL || (articles_dir == NULL && markdown_path == NULL)) {
    help_args(argv[0]);
    return 0;
  }

  if (markdown_path != NULL)
    from_markdown(markdown_path, config_path);

  if (articles_dir != NULL)
    articles_index(articles_dir, config_path);

  return 0;
}
