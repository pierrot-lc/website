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
  node->content = content;
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

void free_tree(Node *root) {
  if (root->content == NULL)
    free(root->content);

  for (int i = 0; i < root->child_count; i++)
    free_tree(root->children[i]);

  free(root);
}

void _print_node(Node *node, unsigned int offset) {
  for (int i = 0; i < offset; i++)
    printf(" ");

  if (node->content != NULL)
    printf("%s ", node->content);

  printf("(%u)\n", node->code);

  for (int i = 0; i < node->child_count; i++)
    _print_node(node->children[i], offset + 2);
}

void print_tree(Node *root) { _print_node(root, 0); }
