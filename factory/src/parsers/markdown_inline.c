#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "parsers/markdown_inline.h"
#include "tree.h"
#include "ts_utils.h"

const TSLanguage *tree_sitter_markdown_inline(void);

static Node *inline_text(const char *, TSNode);
static Node *next_node(const char *, TSNode);

static Node *backslash_escape(const char *source, TSNode ts_node) {
  Node *node;
  unsigned int start, end;

  start = ts_node_start_byte(ts_node) + 1;
  end = ts_node_end_byte(ts_node);
  node = create_node(HASH_TEXT, extract_text(source, start, end));
  return node;
}

static Node *emphasis(const char *source, TSNode ts_node) {
  Node *node, *child;
  TSNode ts_child;
  char *text;
  unsigned int start, end;

  node = create_node(hash(ts_node_type(ts_node)), NULL);

  start = ts_node_start_byte(ts_node);

  // NOTE: The first and last children are the emphasis delimiter.
  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    ts_child = ts_node_named_child(ts_node, i);
    end = ts_node_start_byte(ts_child);

    if (start != end) {
      text = extract_text(source, start, end);
      child = create_node(HASH_TEXT, text);
      add_child(node, child);
    }

    if (hash(ts_node_type(ts_child)) != HASH_EMPHASIS_DELIMITER &&
        hash(ts_node_type(ts_child)) != HASH_CODE_SPAN_DELIMITER) {
      child = next_node(source, ts_child);
      add_child(node, child);
    }

    start = ts_node_end_byte(ts_child);
  }

  return node;
}

static Node *full_reference_link(const char *source, TSNode ts_node) {
  Node *node, *text, *label;

  node = create_node(HASH_LINK, NULL);

  text = inline_text(source, ts_node_named_child(ts_node, 0));
  text->code = HASH_LINK_TEXT;
  add_child(node, text);

  label = create_node(HASH_LINK_LABEL,
                      ts_node_text(source, ts_node_named_child(ts_node, 1)));
  add_child(node, label);

  return node;
}

static Node *image(const char *source, TSNode ts_node) {
  Node *node = NULL, *url = NULL, *desc = NULL;
  TSNode ts_child;

  node = create_node(HASH_IMAGE, NULL);

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    ts_child = ts_node_named_child(ts_node, i);

    switch (hash(ts_node_type(ts_child))) {
    case HASH_IMAGE_DESCRIPTION:
      desc =
          create_node(HASH_IMAGE_DESCRIPTION, ts_node_text(source, ts_child));
      break;

    case HASH_LINK_DESTINATION:
      url = create_node(HASH_LINK_DESTINATION, ts_node_text(source, ts_child));
      break;

    case HASH_LINK_LABEL:
      url = create_node(HASH_LINK_LABEL, ts_node_text(source, ts_child));
      break;

    default:
      fprintf(stderr, "[INLINE IMAGE] Unknown type: %s (%u)",
              ts_node_type(ts_child), hash(ts_node_type(ts_child)));
    }
  }

  if (url != NULL)
    add_child(node, url);

  if (desc != NULL)
    add_child(node, desc);

  return node;
}

static Node *inline_text(const char *source, TSNode ts_node) {
  char *text;
  unsigned int start, end;

  Node *node, *child;
  TSNode ts_child;

  node = create_node(HASH_INLINE, NULL);
  start = ts_node_start_byte(ts_node);

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    ts_child = ts_node_named_child(ts_node, i);

    end = ts_node_start_byte(ts_child);
    text = extract_text(source, start, end);
    child = create_node(HASH_TEXT, text);
    add_child(node, child);

    child = next_node(source, ts_child);
    add_child(node, child);

    start = ts_node_end_byte(ts_child);
  }

  end = ts_node_end_byte(ts_node);
  text = extract_text(source, start, end);
  child = create_node(HASH_TEXT, text);
  add_child(node, child);

  return node;
}

static Node *inline_link(const char *source, TSNode ts_node) {
  Node *node, *text, *destination;
  TSNode child;

  node = create_node(HASH_LINK, NULL);

  for (int i = 0; i < ts_node_named_child_count(ts_node); i++) {
    child = ts_node_named_child(ts_node, i);

    switch (hash(ts_node_type(child))) {
    case HASH_LINK_TEXT:
      text = inline_text(source, child);
      text->code = HASH_LINK_TEXT;
      add_child(node, text);
      break;

    case HASH_LINK_DESTINATION:
      destination =
          create_node(HASH_LINK_DESTINATION, ts_node_text(source, child));
      add_child(node, destination);
      break;

    default:
      fprintf(stderr, "[INLINE LINK] Unknown type: %s (%u)",
              ts_node_type(child), hash(ts_node_type(child)));
      assert(false);
    }
  }

  return node;
}

static Node *latex_block(const char *source, TSNode ts_node) {
  char *delimiter;
  unsigned int start, end, latex_type;

  Node *node;
  TSNode delimiter_start, delimiter_end;

  assert(ts_node_named_child_count(ts_node) == 2);
  delimiter_start = ts_node_named_child(ts_node, 0);
  delimiter_end = ts_node_named_child(ts_node, 1);
  assert(hash(ts_node_type(delimiter_start)) == HASH_LATEX_SPAN_DELIMITER);
  assert(hash(ts_node_type(delimiter_end)) == HASH_LATEX_SPAN_DELIMITER);

  delimiter = ts_node_text(source, delimiter_start);
  if (strcmp(delimiter, "$$") == 0)
    latex_type = HASH_LATEX_DISPLAY;
  else if (strcmp(delimiter, "$") == 0)
    latex_type = HASH_LATEX_INLINE;
  else
    assert(false);
  free(delimiter);

  start = ts_node_end_byte(delimiter_start);
  end = ts_node_start_byte(delimiter_end);
  node = create_node(latex_type, extract_text(source, start, end));
  return node;
}

static Node *shortcut_link(const char *source, TSNode ts_node) {
  Node *node, *text;

  node = create_node(HASH_LINK, NULL);

  text = inline_text(source, ts_node_named_child(ts_node, 0));
  text->code = HASH_LINK_TEXT;
  add_child(node, text);

  return node;
}

static Node *uri_autolink(const char *source, TSNode ts_node) {
  Node *node, *text, *inline_text, *destination;
  char *url;

  node = create_node(HASH_LINK, NULL);

  text = create_node(HASH_LINK_TEXT, NULL);
  url = extract_text(source, ts_node_start_byte(ts_node) + 1,
                     ts_node_end_byte(ts_node) - 1);
  inline_text = create_node(HASH_TEXT, url);
  add_child(node, text);
  add_child(text, inline_text);

  // Alloc a second time the string, to avoid freeing twice the same string
  // when freeing the tree.
  url = extract_text(source, ts_node_start_byte(ts_node) + 1,
                     ts_node_end_byte(ts_node) - 1);
  destination = create_node(HASH_LINK_DESTINATION, url);
  add_child(node, destination);
  return node;
}

static Node *next_node(const char *source, TSNode ts_node) {
  Node *node;
  switch (hash(ts_node_type(ts_node))) {

  case HASH_BACKSLASH_ESCAPE:
    node = backslash_escape(source, ts_node);
    break;

  case HASH_COLLAPSED_REFERENCE_LINK:
    node = inline_link(source, ts_node);
    break;

  case HASH_CODE_SPAN:
  case HASH_EMPHASIS:
  case HASH_STRONG_EMPHASIS:
    node = emphasis(source, ts_node);
    break;

  case HASH_FULL_REFERENCE_LINK:
    node = full_reference_link(source, ts_node);
    break;

  case HASH_IMAGE:
    node = image(source, ts_node);
    break;

  case HASH_INLINE:
    node = inline_text(source, ts_node);
    break;

  case HASH_INLINE_LINK:
    node = inline_link(source, ts_node);
    break;

  case HASH_LATEX_BLOCK:
    node = latex_block(source, ts_node);
    break;

  case HASH_SHORTCUT_LINK:
    node = shortcut_link(source, ts_node);
    break;

  case HASH_URI_AUTOLINK:
    node = uri_autolink(source, ts_node);
    break;

  default:
    fprintf(stderr, "[TREE MD INLINE] Unknown type: %s (%u)\n",
            ts_node_type(ts_node), hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

/*
 * *Main*
 */

Node *parse_markdown_inline(const char *source) {
  Node *root;
  TSNode ts_root;
  TSTree *ts_tree;
  unsigned int hash_inline;

  ts_tree = ts_parse(source, tree_sitter_markdown_inline());
  ts_root = ts_tree_root_node(ts_tree);
  hash_inline = hash(ts_node_type(ts_root));

  assert(hash_inline == HASH_INLINE);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = next_node(source, ts_root);
  ts_tree_delete(ts_tree);
  return root;
}
