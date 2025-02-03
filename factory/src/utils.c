#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

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

char *node_text(const char *source_code, TSNode node) {
  unsigned int start = ts_node_start_byte(node);
  unsigned int end = ts_node_end_byte(node);
  unsigned int length = end - start;

  char *node_text = malloc(length + 1);
  strncpy(node_text, source_code + start, length);
  node_text[length] = '\0';
  return node_text;
}
