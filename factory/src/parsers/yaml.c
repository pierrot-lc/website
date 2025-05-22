#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "tree.h"
#include "ts_utils.h"

const TSLanguage *tree_sitter_yaml(void);

static Node *block_mapping_pair(const char *source, TSNode ts_node);
static Node *next_node(const char *source, TSNode ts_node);
static void search_values(const char *source, TSNode ts_node, Node *key);

/**
 * Go through all children nodes, parse and add them to the parent node.
 */
static void children(Node *parent, const char *source, TSNode ts_node) {
  Node *child;
  TSNode ts_child;

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    ts_child = ts_node_named_child(ts_node, i);
    child = next_node(source, ts_child);
    if (child != NULL)
      add_child(parent, child);
  }
}

/**
 * Join lines, remove heading ">-".
 */
static Node *block_scalar(const char *source, TSNode ts_node) {
  char *text, *p;

  text = ts_node_text(source, ts_node);

  assert(text[0] == '>' && text[1] == '-');

  // Remove the starting ">-" characters.
  memmove(text, text + 2, strlen(text + 2) + 1);

  // Replace "\n  " by " ".
  while ((p = strstr(text, "\n  ")) != NULL)
    memmove(p, p + 2, strlen(p + 2) + 1);

  // Remove the starting " ".
  memmove(text, text + 1, strlen(text + 1) + 1);

  return create_node(HASH_VALUE, text);
}

/**
 * Add all child values to key node.
 */
static void search_values(const char *source, TSNode ts_node, Node *key) {
  Node *value;

  switch (hash(ts_node_type(ts_node))) {
  case HASH_BLOCK_MAPPING_PAIR:
    value = block_mapping_pair(source, ts_node);
    add_child(key, value);
    break;

  case HASH_BLOCK_SEQUENCE_ITEM:
    value = create_node(HASH_BLOCK_SEQUENCE_ITEM, NULL);
    add_child(key, value);

    for (int i = 0; i < ts_node_named_child_count(ts_node); i++)
      search_values(source, ts_node_named_child(ts_node, i), value);
    break;

  case HASH_PLAIN_SCALAR:
    value = create_node(HASH_VALUE, ts_node_text(source, ts_node));
    add_child(key, value);
    break;

  case HASH_BLOCK_SCALAR:
    value = block_scalar(source, ts_node);
    add_child(key, value);
    break;

  default:
    for (int i = 0; i < ts_node_named_child_count(ts_node); i++)
      search_values(source, ts_node_named_child(ts_node, i), key);
  }
}

static Node *block_mapping_pair(const char *source, TSNode ts_node) {
  char *text;

  Node *key;

  text = ts_node_text(source, ts_node_named_child(ts_node, 0));
  key = create_node(HASH_KEY, text);
  search_values(source, ts_node_named_child(ts_node, 1), key);
  return key;
}

static Node *next_node(const char *source, TSNode ts_node) {
  Node *node;

  switch (hash(ts_node_type(ts_node))) {
  case HASH_BLOCK_MAPPING_PAIR:
    node = block_mapping_pair(source, ts_node);
    break;

  case HASH_BLOCK_MAPPING:
  case HASH_BLOCK_NODE:
  case HASH_DOCUMENT:
  case HASH_STREAM:
    node = create_node(hash(ts_node_type(ts_node)), NULL);
    children(node, source, ts_node);
    break;

  default:
    fprintf(stderr, "[YAML] Unknown hash: %s (%u)\n", ts_node_type(ts_node),
            hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

Node *get_key(Node *node, char *key) {
  Node *node_key;

  if (node->child_count == 0)
    return NULL;

  if (node->code == HASH_KEY && strcmp(node->content, key) == 0)
    return node;

  for (int i = 0; i < node->child_count; i++) {
    node_key = get_key(node->children[i], key);
    if (node_key != NULL)
      return node_key;
  }

  return NULL;
}

char *check_scalar_value(Node *key) {
  if (key == NULL) {
    perror("Key not found");
    exit(EXIT_FAILURE);
  }

  if (key->code != HASH_KEY) {
    perror("Arg is not a key");
    exit(EXIT_FAILURE);
  }

  if (key->child_count == 0) {
    perror("Key don't have any child value");
    exit(EXIT_FAILURE);
  }

  if (key->child_count > 1) {
    perror("Key have multiple children values");
    exit(EXIT_FAILURE);
  }

  if (key->children[0]->content == NULL) {
    perror("Value is null");
    exit(EXIT_FAILURE);
  }

  return key->children[0]->content;
}

Node *parse_yaml(const char *source) {
  unsigned int hash_stream;

  Node *root;
  TSNode ts_root;
  TSTree *ts_tree;

  ts_tree = ts_parse(source, tree_sitter_yaml());
  ts_root = ts_tree_root_node(ts_tree);
  hash_stream = hash(ts_node_type(ts_root));

  assert(hash_stream == HASH_STREAM);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = create_node(hash_stream, NULL);
  children(root, source, ts_root);
  ts_tree_delete(ts_tree);

  return root;
}
