#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hash.h"
#include "tree.h"

Node *create_node(unsigned int code, char *content) {
  Node *node = (Node *)malloc(sizeof(Node));
  assert(node != NULL);

  node->code = code;
  node->data.content = content;
  node->children = NULL;
  node->child_count = 0;
  node->parent = NULL;
  return node;
}

void add_child(Node *parent, Node *child) {
  parent->children = (Node **)realloc(
      parent->children, (parent->child_count + 1) * sizeof(Node *));
  assert(parent->children != NULL);

  parent->children[parent->child_count] = child;
  parent->child_count++;
  child->parent = parent;
}

Node *tree_root(Node *node) {
  if (node->parent == NULL)
    return node;

  return tree_root(node->parent);
}

void _free_table(Node *node) {
  Table *table;
  assert(node->code == HASH_PIPE_TABLE);

  table = node->data.table;

  for (int j = 0; j < table->ncols; j++) {
    free_tree(table->columns[j]->cell);
    free(table->columns[j]);
  }
  free(table->columns);

  for (int i = 0; i < table->nrows; i++) {
    for (int j = 0; j < table->ncols; j++)
      free_tree(table->cells[i][j]);
    free(table->cells[i]);
  }
  free(table->cells);

  free(table);
}

void free_tree(Node *root) {
  switch (root->code) {
  case HASH_PIPE_TABLE:
    _free_table(root);
    break;
  default:
    if (root->data.content == NULL)
      free(root->data.content);
    break;
  }

  for (int i = 0; i < root->child_count; i++)
    free_tree(root->children[i]);

  free(root);
}

void _print_node(Node *node, unsigned int offset) {
  for (int i = 0; i < offset; i++)
    fprintf(stderr, " ");

  if (node->data.content != NULL)
    fprintf(stderr, "%s ", node->data.content);

  fprintf(stderr, "(%u)\n", node->code);

  for (int i = 0; i < node->child_count; i++)
    _print_node(node->children[i], offset + 2);
}

void print_tree(Node *root) { _print_node(root, 0); }

Node *search_node(Node *node, unsigned int hash_code) {
  Node *found;

  if (node->code == hash_code)
    return node;

  for (int i = 0; i < node->child_count; i++) {
    found = search_node(node->children[i], hash_code);
    if (found != NULL)
      return found;
  }

  return NULL;
}
