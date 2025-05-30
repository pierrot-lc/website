#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parsers/markdown.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"
#include "writers/body.h"
#include "writers/head.h"

int from_markdown(char *markdown_path, char *config_path) {
  char *source;
  char file_path[256];

  Node *yaml, *markdown, *header;

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

  if ((header = get_key(markdown, "header-file")) != NULL) {
    snprintf(file_path, 256, "%s/%s", dirname(config_path),
             get_value_scalar(header));

    source = read_file(file_path);
    if (source == NULL) {
      fprintf(stderr, "Can't open %s\n", file_path);
      return -1;
    }
    header = parse_markdown(source);
    free(source);
  }

  // Generate the page.
  fprintf(stdout, "<!DOCTYPE html>\n");
  fprintf(stdout, "<html>\n");
  write_head(stdout, markdown);
  fprintf(stdout, "<body>\n");

  if (header != NULL) {
    fprintf(stdout, "<header>\n");
    write_tree(stdout, header);
    fprintf(stdout, "</header>\n");
  }

  fprintf(stdout, "<main>\n");
  write_page_info(stdout, markdown);
  write_tree(stdout, markdown);
  fprintf(stdout, "</main>\n");

  fprintf(stdout, "</body>\n");
  fprintf(stdout, "</html>\n");

  free_tree(markdown);

  return 0;
}

void help_args(char *self) {
  fprintf(stdout, "Usage: %s --config <config-path> --md <md-path>\n", self);
}

int main(int argc, char *argv[]) {
  char *markdown_path = NULL, *config_path = NULL;

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

    help_args(argv[0]);
    return 0;
  }

  if (config_path == NULL || markdown_path == NULL) {
    help_args(argv[0]);
    return 0;
  }

  return from_markdown(markdown_path, config_path);
}
