#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <tree_sitter/api.h>

#include "hash.h"
#include "parsers/markdown.h"
#include "parsers/markdown_inline.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"

const TSLanguage *tree_sitter_markdown(void);

static Node *next_node(const char *, TSNode);

/**
 * Go through all children nodes, parse and add them to the parent node.
 */
static void children(Node *parent, const char *source, TSNode ts_parent) {
  Node *child;
  TSNode ts_child;

  for (int i = 0; i < ts_node_named_child_count(ts_parent); i++) {
    ts_child = ts_node_named_child(ts_parent, i);
    child = next_node(source, ts_child);
    if (child != NULL)
      add_child(parent, child);
  }
}

static Node *blockquote(const char *source, TSNode ts_node) {
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

  if (parsed_text->data.content != NULL)
    free(parsed_text->data.content);
  if (parsed_text->children[0]->data.content != NULL)
    free(parsed_text->children[0]->data.content);

  free(parsed_text->children[0]);
  free(parsed_text);

  return node;
}

static Node *code_block(const char *source, TSNode ts_node) {
  Node *node, *content;
  TSNode ts_language, ts_content;

  ts_language = ts_search(ts_node, HASH_LANGUAGE);
  ts_content = ts_search(ts_node, HASH_CODE_FENCE_CONTENT);
  assert(!ts_node_is_null(ts_content));

  node = create_node(HASH_FENCED_CODE_BLOCK, NULL);
  if (!ts_node_is_null(ts_language))
    node->data.content = ts_node_text(source, ts_language);

  content =
      create_node(HASH_CODE_FENCE_CONTENT, ts_node_text(source, ts_content));
  add_child(node, content);

  return node;
}

static Node *heading(const char *source, TSNode ts_node) {
  assert(ts_node_named_child_count(ts_node) == 2);

  Node *node, *child;
  TSNode ts_marker, ts_content;
  char *content;

  const char *h1 = "h1";
  const char *h2 = "h2";
  const char *h3 = "h3";
  const char *h4 = "h4";
  const char *h;

  ts_marker = ts_node_child(ts_node, 0);
  switch (hash(ts_node_type(ts_marker))) {
  case HASH_ATX_H1_MARKER:
    h = h1;
    break;
  case HASH_ATX_H2_MARKER:
    h = h2;
    break;
  case HASH_ATX_H3_MARKER:
    h = h3;
    break;
  case HASH_ATX_H4_MARKER:
    h = h4;
    break;
  default:
    assert(false);
  }

  content = strdup(h);
  node = create_node(HASH_ATX_HEADING, content);

  ts_content = ts_node_child(ts_node, 1);
  child = next_node(source, ts_content);
  if (child != NULL)
    add_child(node, child);
  return node;
}

static Node *html_block(const char *source, TSNode ts_node) {
  char *content;

  Node *node;

  // Ignore last empty line of the block.
  content = extract_text(source, ts_node_start_byte(ts_node),
                         ts_node_end_byte(ts_node) - 1);
  node = create_node(HASH_HTML_BLOCK, content);
  return node;
}

static Node *link_reference_definition(const char *source, TSNode ts_node) {
  Node *node, *child;
  char *label, *destination;

  label = ts_node_text(source, ts_node_named_child(ts_node, 0));
  node = create_node(HASH_LINK_REFERENCE_DEFINITION, label);

  destination = ts_node_text(source, ts_node_named_child(ts_node, 1));
  child = create_node(HASH_LINK_DESTINATION, destination);
  add_child(node, child);

  return node;
}

static Node *list_item(const char *source, TSNode ts_node) {
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

static Node *markdown_inline(const char *source, TSNode ts_node) {
  Node *node;
  char *source_inline;

  source_inline = ts_node_text(source, ts_node);
  node = parse_markdown_inline(source_inline);
  free(source_inline);
  return node;
}

static Node *minus_metadata(const char *source, TSNode ts_node) {
  Node *node, *child;
  char *source_yaml;

  node = create_node(HASH_MINUS_METADATA, NULL);

  source_yaml = ts_node_text(source, ts_node);
  child = parse_yaml(source_yaml);
  free(source_yaml);
  add_child(node, child);

  return node;
}

static Node *pipe_table_cell(const char *source, TSNode ts_node) {
  unsigned long start;
  int end;
  char *source_inline;
  Node *node, *node_inline;

  const char *empty_cell = "&nbsp;";

  node = create_node(HASH_PIPE_TABLE_CELL, NULL);
  assert(ts_node_named_child_count(ts_node) == 0);
  source_inline = ts_node_text(source, ts_node);

  // Remove leading and ending spaces of the cell. Those spaces are typically
  // used for pretty formatting within markdown.
  for (start = 0; start < strlen(source_inline); start++)
    if (source_inline[start] != ' ')
      break;
  for (end = (int)strlen(source_inline) - 1; end >= start; end--)
    if (source_inline[end] != ' ')
      break;
  source_inline[end + 1] = '\0';

  if (strlen(source_inline + start) == 0) {
    free(source_inline);
    source_inline = strdup(empty_cell);
    start = 0;
  }

  node_inline = parse_markdown_inline(source_inline + start);
  free(source_inline);

  add_child(node, node_inline);
  return node;
}

static Node *pipe_table(const char *source, TSNode ts_node) {
  Node *node;
  TSNode ts_headers, ts_delimiters, ts_row, ts_child;
  Table *table;

  const char *center = "center";
  const char *left = "left";
  const char *right = "right";
  const char *unknown = "unknown";
  const char *a;


  // At least headers and delimiters.
  assert(ts_node_named_child_count(ts_node) >= 2);
  table = (Table *)calloc(1, sizeof(Table));

  node = create_node(HASH_PIPE_TABLE, NULL);
  ts_headers = ts_node_named_child(ts_node, 0);
  ts_delimiters = ts_node_named_child(ts_node, 1);
  assert(hash(ts_node_type(ts_headers)) == HASH_PIPE_TABLE_HEADER);
  assert(hash(ts_node_type(ts_delimiters)) == HASH_PIPE_TABLE_DELIMITER_ROW);

  table->ncols = ts_node_named_child_count(ts_headers);
  assert(ts_node_named_child_count(ts_delimiters) == table->ncols);
  table->columns = (Column **)calloc(table->ncols, sizeof(Column *));
  for (int j = 0; j < table->ncols; j++) {
    table->columns[j] = (Column *)calloc(1, sizeof(Column));

    // Column content.
    ts_child = ts_node_named_child(ts_headers, j);
    table->columns[j]->cell = pipe_table_cell(source, ts_child);

    // Column alignment.
    ts_child = ts_node_named_child(ts_delimiters, j);
    assert(hash(ts_node_type(ts_child)) == HASH_PIPE_TABLE_DELIMITER_CELL);
    switch (ts_node_named_child_count(ts_child)) {
    case 0:
      a = unknown;
      break;
    case 1:
      ts_child = ts_node_named_child(ts_child, 0);
      switch (hash(ts_node_type(ts_child))) {
      case HASH_PIPE_TABLE_ALIGN_LEFT:
        a = left;
        break;
      case HASH_PIPE_TABLE_ALIGN_RIGHT:
        a = right;
        break;
      default:
        assert(false);
      }
      break;
    case 2:
      a = center;
      assert(hash(ts_node_type(ts_node_named_child(ts_child, 0))) ==
             HASH_PIPE_TABLE_ALIGN_LEFT);
      assert(hash(ts_node_type(ts_node_named_child(ts_child, 1))) ==
             HASH_PIPE_TABLE_ALIGN_RIGHT);
      break;
    default:
      assert(false);
    }
    strncpy(table->columns[j]->alignment, a, 20);
  }

  // Table content.
  table->nrows = ts_node_named_child_count(ts_node) - 2;
  table->cells = (Node ***)calloc(table->nrows, sizeof(Node **));
  for (int i = 0; i < table->nrows; i++) {
    ts_row = ts_node_named_child(ts_node, i + 2);
    assert(ts_node_named_child_count(ts_row) == table->ncols);

    table->cells[i] = (Node **)calloc(table->ncols, sizeof(Node *));

    for (int j = 0; j < table->ncols; j++) {
      ts_child = ts_node_named_child(ts_row, j);
      table->cells[i][j] = pipe_table_cell(source, ts_child);
    }
  }

  node->data.table = table;
  return node;
}

/**
 * Choose the right parsing function to use.
 */
static Node *next_node(const char *source, TSNode ts_node) {
  Node *node = NULL;
  switch (hash(ts_node_type(ts_node))) {
  case HASH_ATX_HEADING:
    node = heading(source, ts_node);
    break;

  case HASH_BLOCK_QUOTE:
    node = blockquote(source, ts_node);
    break;

  case HASH_FENCED_CODE_BLOCK:
    node = code_block(source, ts_node);
    break;

  case HASH_HTML_BLOCK:
    node = html_block(source, ts_node);
    break;

  case HASH_INLINE:
    node = markdown_inline(source, ts_node);
    break;

  case HASH_LINK_REFERENCE_DEFINITION:
    node = link_reference_definition(source, ts_node);
    break;

  case HASH_LIST_ITEM:
    node = list_item(source, ts_node);
    break;

  case HASH_MINUS_METADATA:
    node = minus_metadata(source, ts_node);
    break;

  case HASH_PIPE_TABLE:
    node = pipe_table(source, ts_node);
    break;

  // Trivial nodes.
  case HASH_BLOCK_CONTINUATION:
  case HASH_BLOCK_QUOTE_MARKER:
  case HASH_LIST:
  case HASH_PARAGRAPH:
  case HASH_SECTION:
  case HASH_THEMATIC_BREAK:
    node = create_node(hash(ts_node_type(ts_node)), NULL);
    children(node, source, ts_node);
    break;

  default:
    fprintf(stderr, "[TREE MD] Unknown hash: %s (%u)\n", ts_node_type(ts_node),
            hash(ts_node_type(ts_node)));
    assert(false);
  }

  return node;
}

Node *search_label_destination(Node *node, char *label) {
  Node *found;

  if (node->code == HASH_LINK_REFERENCE_DEFINITION &&
      strcmp(node->data.content, label) == 0)
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

  ts_tree = ts_parse(source, tree_sitter_markdown());
  ts_root = ts_tree_root_node(ts_tree);
  hash_document = hash(ts_node_type(ts_root));

  assert(hash_document == HASH_DOCUMENT);
  assert(ts_node_is_null(ts_node_next_sibling(ts_root)));

  root = create_node(hash_document, NULL);
  children(root, source, ts_root);
  ts_tree_delete(ts_tree);
  return root;
}
