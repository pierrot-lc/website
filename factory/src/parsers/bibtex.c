#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "parsers/bibtex.h"
#include "tree.h"
#include "ts_utils.h"

const TSLanguage *tree_sitter_bibtex(void);

static Node *next_node(const char *, TSNode);

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

static Node *field(const char *source, TSNode ts_node) {
  char *text;
  Node *node, *name, *value;
  TSNode ts_name, ts_value;

  ts_name = ts_node_named_child(ts_node, 0);
  ts_value = ts_node_named_child(ts_node, 1);
  assert(hash(ts_node_type(ts_name)) == HASH_IDENTIFIER);
  assert(hash(ts_node_type(ts_value)) == HASH_VALUE);
  assert(source[ts_node_start_byte(ts_value)] == '{' &&
         source[ts_node_end_byte(ts_value) - 1] == '}');

  node = create_node(HASH_FIELD, strdup("field"));
  name = create_node(HASH_IDENTIFIER, ts_node_text(source, ts_name));
  text = extract_text(source, ts_node_start_byte(ts_value) + 1,
                      ts_node_end_byte(ts_value) - 1);
  value = create_node(HASH_VALUE, text);
  add_child(node, name);
  add_child(node, value);
  return node;
}

static Node *key(const char *source, TSNode ts_node) {
  Node *node;
  node = create_node(HASH_KEY_BRACE, ts_node_text(source, ts_node));
  return node;
}

static Node *entry_type(const char *source, TSNode ts_node) {
  char *text;
  Node *node, *value;

  assert(source[ts_node_start_byte(ts_node)] == '@');

  node = create_node(HASH_ENTRY_TYPE, strdup("bibtex-entry-type"));
  text = extract_text(source, ts_node_start_byte(ts_node) + 1,
                      ts_node_end_byte(ts_node));
  value = create_node(HASH_VALUE, text);
  add_child(node, value);
  return node;
}

static Node *next_node(const char *source, TSNode ts_node) {
  Node *node = NULL;

  switch (hash(ts_node_type(ts_node))) {
  case HASH_FIELD:
    node = field(source, ts_node);
    break;

  case HASH_KEY_BRACE:
    node = key(source, ts_node);
    break;

  case HASH_ENTRY_TYPE:
    node = entry_type(source, ts_node);
    break;

  case HASH_ENTRY:
    node = create_node(hash(ts_node_type(ts_node)), NULL);
    children(node, source, ts_node);
    break;

  default:
    fprintf(stderr, "[BIBTEX] Unknown node: %s (%u)\n", ts_node_type(ts_node),
            hash(ts_node_type(ts_node)));
    assert(false);
  }
  return node;
}

Node *get_field(Node *node, char *name) {
  Node *field;

  if (node->child_count == 0)
    return NULL;

  if (node->code == HASH_FIELD && strcmp(node->children[0]->content, name) == 0)
    return node;

  for (int i = 0; i < node->child_count; i++) {
    field = get_field(node->children[i], name);
    if (field != NULL)
      return field;
  }

  return NULL;
}

Node *search_bibliography_entry(Node *node, char *entry) {
  Node *found;

  if (node->code == HASH_KEY_BRACE && strcmp(node->content, entry) == 0)
    return node->parent;

  for (int i = 0; i < node->child_count; i++) {
    found = search_bibliography_entry(node->children[i], entry);
    if (found != NULL)
      return found;
  }

  return NULL;
}

Author *parse_authors(Node *entry) {
  char *token, *saveptr, *author;
  Author *list, *curr = NULL;
  Node *node = get_field(entry, "author");

  if (node == NULL) {
    fprintf(stderr, "'author' not found for bibtex entry %s\n",
            entry->children[1]->content);
    assert(false);
  }

  author = strdup(node->children[1]->content);
  token = strtok_r(author, " ", &saveptr);

  curr = list = (Author *)calloc(1, sizeof(Author));

  while (token != NULL) {
    if (strcmp(token, "and") == 0) {
      token = strtok_r(NULL, " ", &saveptr);
      curr->next = (Author *)calloc(1, sizeof(Author));
      curr = curr->next;
      continue;
    }

    if (token[strlen(token) - 1] == ',') {
      token[strlen(token) - 1] = '\0';
      strcpy(curr->lastname, token);
    } else if (curr->firstname[0] != '\0')
      strcpy(curr->lastname, token);
    else
      strcpy(curr->firstname, token);

    token = strtok_r(NULL, " ", &saveptr);
  }

  return list;
}

void mark_cited(Node *entry) {
  Node *node, *name, *value;

  node = create_node(HASH_FIELD, strdup("field"));
  name = create_node(HASH_IDENTIFIER, strdup("cited?"));
  value = create_node(HASH_VALUE, strdup("yes"));
  add_child(node, name);
  add_child(node, value);
  add_child(entry, node);
}

Node *parse_bibtex(const char *source) {
  Node *root;
  TSNode ts_root;
  TSTree *ts_tree;

  ts_tree = ts_parse(source, tree_sitter_bibtex());
  ts_root = ts_tree_root_node(ts_tree);

  assert(hash(ts_node_type(ts_root)) == HASH_DOCUMENT);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = create_node(HASH_BIBTEX_DOCUMENT, NULL);
  children(root, source, ts_root);
  ts_tree_delete(ts_tree);
  return root;
}
