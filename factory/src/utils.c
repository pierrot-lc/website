#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "utils.h"

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

char *node_text(const char *source, TSNode node) {
  return extract_text(source, ts_node_start_byte(node), ts_node_end_byte(node));
}

TSNode search_root(TSNode node) {
  TSNode parent = ts_node_parent(node);
  if (ts_node_is_null(parent))
    return parent;

  return search_root(parent);
}

TSNode search_node(const char *source, TSNode node, unsigned int hash_value,
                   char *text_value) {
  // NOTE: A failed search is returning the null node. The issue is that I
  // don't know how to instanciate a null node directly, so I use the fact that
  // the next_named_sibling is returning the null node when running out of
  // siblings. To have a coherent search, I start by searching for the target
  // node in the children, and then in the siblings.

  // Check children.
  if (ts_node_named_child_count(node) >= 1) {
    TSNode found = search_node(source, ts_node_named_child(node, 0), hash_value,
                               text_value);
    if (!ts_node_is_null(found))
      return found;
  }

  // Check current node and its siblings.
  do {
    if (hash(ts_node_type(node)) == hash_value) {
      char *text = node_text(source, node);
      if (strcmp(text, text_value) == 0) {
        free(text);
        return node;
      }

      free(text);
    }
    node = ts_node_next_named_sibling(node);
  } while (!ts_node_is_null(node));

  return node; // It is the null node at this point.
}
