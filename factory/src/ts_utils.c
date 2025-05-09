#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "ts_utils.h"

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

char *extract_text(const char *buffer, unsigned int start, unsigned int end) {
  assert(end >= start);
  unsigned int length = end - start;
  char *text = (char *)malloc(length + 1);

  strncpy(text, buffer + start, length);
  text[length] = '\0';
  return text;
}

char *ts_node_text(const char *source, TSNode node) {
  return extract_text(source, ts_node_start_byte(node), ts_node_end_byte(node));
}

TSNode ts_tree_root(TSNode node) {
  TSNode parent = ts_node_parent(node);
  if (ts_node_is_null(parent))
    return node;

  return ts_tree_root(parent);
}
