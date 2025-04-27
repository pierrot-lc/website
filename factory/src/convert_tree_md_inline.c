#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "convert_tree_md_inline.h"
#include "hash.h"
#include "tree.h"
#include "utils.h"

static Node *_inline(const char *, TSNode);
static Node *_node(const char *, TSNode);

/*
 * *Converters*
 */

static Node *_emph(const char *source, TSNode ts_node) {
  Node *node, *child;
  TSNode ts_child;
  char *text, *emph;
  unsigned int start, end;

  ts_child = ts_node_named_child(ts_node, 0);
  emph = node_text(source, ts_child);

  if (strcmp(emph, "*") == 0)
    node = create_node(HASH_EMPH_STRONG, NULL);
  else if (strcmp(emph, "_") == 0)
    node = create_node(HASH_EMPH_EM, NULL);
  else
    assert(false);

  start = ts_node_end_byte(ts_child);

  // NOTE: The first and last children are the emphasis delimiter.
  for (int i = 1; i < ts_node_named_child_count(ts_node) - 1; i++) {
    ts_child = ts_node_named_child(ts_node, i);

    end = ts_node_start_byte(ts_child);
    text = extract_text(source, start, end);
    child = create_node(HASH_TEXT, text);
    add_child(node, child);

    child = _node(source, ts_child);
    add_child(node, child);

    start = ts_node_end_byte(ts_child);
  }

  ts_child =
      ts_node_named_child(ts_node, ts_node_named_child_count(ts_node) - 1);
  end = ts_node_start_byte(ts_child);
  text = extract_text(source, start, end);
  child = create_node(HASH_TEXT, text);
  add_child(node, child);

  free(emph);

  return node;
}

static Node *_full_reference_link(const char *source, TSNode ts_node) {
  Node *node, *text, *destination;
  TSNode ts_label, ts_destination;
  char *label;

  node = create_node(HASH_LINK, NULL);

  text = _inline(source, ts_node_named_child(ts_node, 0));
  text->code = HASH_LINK_TEXT;
  add_child(node, text);

  // Search for the label destination in the tree.
  label = node_text(source, ts_node_named_child(ts_node, 1));
  ts_label = search_node(source, search_root(ts_node), HASH_LINK_LABEL, label);
  free(label);

  if (!ts_node_is_null(ts_label)) {
    ts_destination = ts_node_next_named_sibling(ts_label);
    if (!ts_node_is_null(ts_destination)) {
      destination =
          create_node(HASH_LINK_DESTINATION, node_text(source, ts_destination));
      add_child(node, destination);
    }
  }

  return node;
}

static Node *_inline(const char *source, TSNode ts_node) {
  Node *node, *child;
  TSNode ts_child;
  char *text;
  unsigned int start, end;

  node = create_node(HASH_INLINE, NULL);
  start = ts_node_start_byte(ts_node);

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    ts_child = ts_node_named_child(ts_node, i);

    end = ts_node_start_byte(ts_child);
    text = extract_text(source, start, end);
    child = create_node(HASH_TEXT, text);
    add_child(node, child);

    child = _node(source, ts_child);
    add_child(node, child);

    start = ts_node_end_byte(ts_child);
  }

  end = ts_node_end_byte(ts_node);
  text = extract_text(source, start, end);
  child = create_node(HASH_TEXT, text);
  add_child(node, child);

  return node;
}

static Node *_inline_link(const char *source, TSNode ts_node) {
  Node *node, *text, *destination;
  TSNode child;

  node = create_node(HASH_LINK, NULL);

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    child = ts_node_named_child(ts_node, i);

    switch (hash(ts_node_type(child))) {
    case HASH_LINK_TEXT:
      text = _inline(source, child);
      text->code = HASH_LINK_TEXT;
      add_child(node, text);
      break;

    case HASH_LINK_DESTINATION:
      destination =
          create_node(HASH_LINK_DESTINATION, node_text(source, child));
      add_child(node, destination);
      break;

    default:
      fprintf(stderr, "[INLINE LINK] Unexpected hash type: %u",
              hash(ts_node_type(child)));
      assert(false);
    }
  }

  return node;
}

static Node *_shortcut_link(const char *source, TSNode ts_node) {
  Node *node, *text;

  node = create_node(HASH_LINK, NULL);

  text = _inline(source, ts_node_named_child(ts_node, 0));
  text->code = HASH_LINK_TEXT;
  add_child(node, text);

  return node;
}

static Node *_uri_autolink(const char *source, TSNode ts_node) {
  Node *node, *text, *destination;

  node = create_node(HASH_LINK, NULL);

  text = create_node(HASH_LINK_TEXT, node_text(source, ts_node));
  add_child(node, text);

  destination = create_node(HASH_LINK_DESTINATION, node_text(source, ts_node));
  add_child(node, destination);
  return node;
}

/*
 * *Utils*
 */

static Node *_node(const char *source, TSNode ts_node) {
  Node *node;
  switch (hash(ts_node_type(ts_node))) {
  case HASH_EMPHASIS:
    node = _emph(source, ts_node);
    break;

  case HASH_FULL_REFERENCE_LINK:
    node = _full_reference_link(source, ts_node);
    break;

  case HASH_INLINE:
    node = _inline(source, ts_node);
    break;

  case HASH_INLINE_LINK:
    node = _inline_link(source, ts_node);
    break;

  case HASH_SHORTCUT_LINK:
    node = _shortcut_link(source, ts_node);
    break;

  case HASH_URI_AUTOLINK:
    node = _uri_autolink(source, ts_node);
    break;

  default:
    fprintf(stderr, "[TREE MD INLINE] Unknown hash: %u",
            hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

/*
 * *Main*
 */

Node *convert_tree_md_inline(const char *source, TSTree *ts_tree) {
  TSNode ts_root;
  unsigned int hash_inline;

  ts_root = ts_tree_root_node(ts_tree);
  hash_inline = hash(ts_node_type(ts_root));

  assert(hash_inline == HASH_INLINE);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  return _node(source, ts_root);
}
