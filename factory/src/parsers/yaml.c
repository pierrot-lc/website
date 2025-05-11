#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "parsers/utils.h"
#include "tree.h"
#include "ts_utils.h"

const TSLanguage *tree_sitter_yaml(void);

static Node *_node(const char *source, TSNode ts_node);

static void search_value(const char *source, TSNode ts_node, Node *key) {
  Node *value;

  if (hash(ts_node_type(ts_node)) == HASH_PLAIN_SCALAR) {
    value = create_node(HASH_VALUE, ts_node_text(source, ts_node));
    add_child(key, value);
    return;
  }

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++)
    search_value(source, ts_node_named_child(ts_node, i), key);
}

static Node *block_mapping_pair(const char *source, TSNode ts_node) {
  char *text;

  Node *key;

  text = ts_node_text(source, ts_node_named_child(ts_node, 0));
  key = create_node(HASH_KEY, text);
  search_value(source, ts_node_named_child(ts_node, 1), key);
  return key;
}

static void _children(Node *parent, const char *source, TSNode ts_node) {
  Node *child;
  TSNode ts_child;

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    ts_child = ts_node_named_child(ts_node, i);
    child = _node(source, ts_child);
    if (child != NULL)
      add_child(parent, child);
  }
}

static Node *_node(const char *source, TSNode ts_node) {
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
    _children(node, source, ts_node);
    break;

  default:
    fprintf(stderr, "[YAML] Unknown hash: %s (%u)\n", ts_node_type(ts_node),
            hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

/*
 * *Main*
 */

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

Node *parse_yaml(const char *source) {
  unsigned int hash_stream;

  Node *root;
  TSNode ts_root;
  TSTree *ts_tree;

  ts_tree = parse(source, tree_sitter_yaml());
  ts_root = ts_tree_root_node(ts_tree);
  hash_stream = hash(ts_node_type(ts_root));

  assert(hash_stream == HASH_STREAM);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = create_node(hash_stream, NULL);
  _children(root, source, ts_root);
  ts_tree_delete(ts_tree);

  return root;
}
