#include "writer.h"
#include "hash.h"
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

/* Read the node's text in the source code based on the starting and ending
 * bytes. You are responsible to `free` the returned string.
 */
char *get_node_text(const char *source_code, TSNode node) {
  unsigned int start = ts_node_start_byte(node);
  unsigned int end = ts_node_end_byte(node);
  unsigned int length = end - start;

  char *node_text = malloc(length + 1);
  strncpy(node_text, source_code + start, length);
  node_text[length] = '\0';
  return node_text;
}

void write_tree(FILE *file, const char *source_code, TSNode node) {
  char *text;

  switch (hash(ts_node_type(node))) {
  case HASH_HEADING:
    fprintf(file, "<h1>");
    write_children(file, source_code, node);
    fprintf(file, "</h1>\n");
    break;

  case HASH_CONTENT:
    fprintf(file, "<p>");
    write_children(file, source_code, node);
    fprintf(file, "</p>\n");
    break;

  case HASH_TEXT:
    text = get_node_text(source_code, node);
    fprintf(file, "%s", text);
    free(text);
    break;

  case HASH_SECTION:
  case HASH_SOURCE_FILE:
    write_children(file, source_code, node);
    break;

  default:
    printf("%s\n", ts_node_type(node));
    write_children(file, source_code, node);
    break;
  }
}

void write_children(FILE *file, const char *source_code, TSNode node) {
  for (int i = 0; i < ts_node_named_child_count(node); i++)
    write_tree(file, source_code, ts_node_named_child(node, i));
}
