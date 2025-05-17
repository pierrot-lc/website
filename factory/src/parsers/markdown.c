#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "parsers/markdown.h"
#include "parsers/markdown_inline.h"
#include "parsers/utils.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"

const TSLanguage *tree_sitter_markdown(void);

static Node *_node(const char *, TSNode);
static void _children(Node *parent, const char *source, TSNode ts_parent);

/*
 * *Converters*
 *
 * Convert TSNode into Node. Each function is specifically made for a certain
 * type of TSNode.
 */

static Node *_blockquote(const char *source, TSNode ts_node) {
  char *text, *p;

  Node *node, *parsed_text;

  node = create_node(HASH_BLOCK_QUOTE, NULL);

  // The blockquote markers are in the way for the inline parsing. When it
  // contains multiple lines, some "> " are inserted. We remove all of them and
  // parse again this part of markdown.
  text = ts_node_text(source, ts_node);

  // Add a "\n" at the beggining to make the first line appear the same as the
  // others.
  text = (char *)realloc(text, strlen(text) + 2);
  memmove(text + 1, text, strlen(text) + 1);
  text[0] = '\n';

  // Replace all "\n> " by "\n>".
  while ((p = strstr(text, "\n> ")) != NULL)
    memmove(p + 2, p + 3, strlen(p + 3) + 1);

  // Replace all "\n>" by "\n".
  while ((p = strstr(text, "\n>")) != NULL)
    memmove(p + 1, p + 2, strlen(p + 2) + 1);

  // Parse the markdown again. Ignore the first "DOCUMENT" and "SECTION" nodes.
  parsed_text = parse_markdown(text);
  free(text);
  assert(parsed_text->code == HASH_DOCUMENT && parsed_text->child_count == 1 &&
         parsed_text->children[0]->code == HASH_SECTION);

  for (int i = 0; i < parsed_text->children[0]->child_count; i++)
    add_child(node, parsed_text->children[0]->children[i]);

  if (parsed_text->content != NULL)
    free(parsed_text->content);
  if (parsed_text->children[0]->content != NULL)
    free(parsed_text->children[0]->content);

  free(parsed_text->children[0]);
  free(parsed_text);

  return node;
}

static Node *_code_block(const char *source, TSNode ts_node) {
  Node *node, *content;
  TSNode ts_language, ts_content;

  ts_language = ts_search(ts_node, HASH_LANGUAGE);
  ts_content = ts_search(ts_node, HASH_CODE_FENCE_CONTENT);
  assert(!ts_node_is_null(ts_content));

  node = create_node(HASH_FENCED_CODE_BLOCK, NULL);
  if (!ts_node_is_null(ts_language))
    node->content = ts_node_text(source, ts_language);

  content =
      create_node(HASH_CODE_FENCE_CONTENT, ts_node_text(source, ts_content));
  add_child(node, content);

  return node;
}

static Node *_heading(const char *source, TSNode ts_node) {
  assert(ts_node_named_child_count(ts_node) == 2);

  Node *node, *child;
  TSNode ts_marker, ts_content;
  char *content;

  const char *h1 = "h1";
  const char *h2 = "h2";
  const char *h;

  ts_marker = ts_node_child(ts_node, 0);
  switch (hash(ts_node_type(ts_marker))) {
  case HASH_ATX_H1_MARKER:
    h = h1;
    break;
  case HASH_ATX_H2_MARKER:
    h = h2;
    break;
  default:
    assert(false);
  }

  content = (char *)malloc((strlen(h) + 1) * sizeof(char));
  strcpy(content, h);
  node = create_node(HASH_ATX_HEADING, content);

  ts_content = ts_node_child(ts_node, 1);
  child = _node(source, ts_content);
  if (child != NULL)
    add_child(node, child);
  return node;
}

static Node *_inline(const char *source, TSNode ts_node) {
  Node *node;
  char *source_inline;

  source_inline = ts_node_text(source, ts_node);
  node = parse_markdown_inline(source_inline);
  free(source_inline);
  return node;
}

static Node *_link_reference_definition(const char *source, TSNode ts_node) {
  Node *node, *child;
  char *label, *destination;

  label = ts_node_text(source, ts_node_named_child(ts_node, 0));
  node = create_node(HASH_LINK_REFERENCE_DEFINITION, label);

  destination = ts_node_text(source, ts_node_named_child(ts_node, 1));
  child = create_node(HASH_LINK_DESTINATION, destination);
  add_child(node, child);

  return node;
}

static Node *_list_item(const char *source, TSNode ts_node) {
  char *source_inline;

  TSNode ts_content;
  Node *node;

  assert(ts_node_named_child_count(ts_node) == 2);

  ts_content = ts_node_named_child(ts_node, 1);
  assert(hash(ts_node_type(ts_content)) == HASH_PARAGRAPH);
  assert(hash(ts_node_type(ts_node_named_child(ts_content, 0))) == HASH_INLINE);

  source_inline = ts_node_text(source, ts_node_named_child(ts_content, 0));
  node = parse_markdown_inline(source_inline);
  free(source_inline);

  return node;
}

static Node *_minus_metadata(const char *source, TSNode ts_node) {
  Node *node, *child;
  char *source_yaml;

  node = create_node(HASH_MINUS_METADATA, NULL);

  source_yaml = ts_node_text(source, ts_node);
  child = parse_yaml(source_yaml);
  free(source_yaml);
  add_child(node, child);

  return node;
}

/*
 * *Utils*
 *
 * The following functions do not directly convert a TSNode into a Node.
 */

/* Choose the right convert function to use. */
static Node *_node(const char *source, TSNode ts_node) {
  Node *node = NULL;
  switch (hash(ts_node_type(ts_node))) {
  case HASH_ATX_HEADING:
    node = _heading(source, ts_node);
    break;

  case HASH_BLOCK_QUOTE:
    node = _blockquote(source, ts_node);
    break;

  case HASH_FENCED_CODE_BLOCK:
    node = _code_block(source, ts_node);
    break;

  case HASH_HTML_BLOCK:
    // TODO
    node = create_node(HASH_PARAGRAPH, NULL);
    break;

  case HASH_INLINE:
    node = _inline(source, ts_node);
    break;

  case HASH_LINK_REFERENCE_DEFINITION:
    node = _link_reference_definition(source, ts_node);
    break;

  case HASH_LIST_ITEM:
    node = _list_item(source, ts_node);
    break;

  case HASH_MINUS_METADATA:
    node = _minus_metadata(source, ts_node);
    break;

  // Trivial nodes.
  case HASH_BLOCK_CONTINUATION:
  case HASH_BLOCK_QUOTE_MARKER:
  case HASH_LIST:
  case HASH_PARAGRAPH:
  case HASH_SECTION:
    node = create_node(hash(ts_node_type(ts_node)), NULL);
    _children(node, source, ts_node);
    break;

  default:
    fprintf(stderr, "[TREE MD] Unknown hash: %s (%u)\n", ts_node_type(ts_node),
            hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

/* Loop over all children of the TSNode and convert them. */
static void _children(Node *parent, const char *source, TSNode ts_parent) {
  Node *child;
  TSNode ts_child;

  for (int i = 0; i < ts_node_named_child_count(ts_parent); i++) {
    ts_child = ts_node_named_child(ts_parent, i);
    child = _node(source, ts_child);
    if (child != NULL)
      add_child(parent, child);
  }
}

/*
 * *Main*
 */

Node *search_label_destination(Node *node, char *label) {
  Node *found;

  if (node->code == HASH_LINK_REFERENCE_DEFINITION &&
      strcmp(node->content, label) == 0)
    return node->children[0];

  for (int i = 0; i < node->child_count; i++) {
    found = search_label_destination(node->children[i], label);
    if (found != NULL)
      return found;
  }

  return NULL;
}

Node *parse_markdown(const char *source) {
  Node *root;
  TSNode ts_root;
  TSTree *ts_tree;
  unsigned int hash_document;

  ts_tree = parse(source, tree_sitter_markdown());
  ts_root = ts_tree_root_node(ts_tree);
  hash_document = hash(ts_node_type(ts_root));

  assert(hash_document == HASH_DOCUMENT);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = create_node(hash_document, NULL);
  _children(root, source, ts_root);
  ts_tree_delete(ts_tree);
  return root;
}
