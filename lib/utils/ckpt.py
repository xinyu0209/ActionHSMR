from typing import List, Dict

def replace_state_dict_name_prefix(state_dict:Dict[str, object], old_prefix:str, new_prefix:str):
    ''' Replace the prefix of the keys in the state_dict. '''
    for old_name in list(state_dict.keys()):
        if old_name.startswith(old_prefix):
            new_name = new_prefix + old_name[len(old_prefix):]
            state_dict[new_name] = state_dict.pop(old_name)

    return state_dict


def match_prefix_and_remove_state_dict(state_dict:Dict[str, object], prefix:str):
    ''' Remove the keys in the state_dict that start with the prefix. '''
    for name in list(state_dict.keys()):
        if name.startswith(prefix):
            state_dict.pop(name)
    return state_dict


class StateDictTree:
    def __init__(self, keys:List[str]):
        self.tree = {}
        for key in keys:
            parts = key.split('.')
            self._recursively_add_leaf(self.tree, parts, key)


    def rich_print(self, depth:int=-1):
        from rich.tree import Tree
        from rich import print
        rich_tree = Tree('.')
        self._recursively_build_rich_tree(rich_tree, self.tree, 0, depth)
        print(rich_tree)

    def update_node_name(self, old_name:str, new_name:str):
        ''' Input full node name and the whole node will be moved to the new name. '''
        old_parts = old_name.split('.')
        # 1. Delete the old node.
        try:
            parent = None
            node = self.tree
            for part in old_parts:
                parent = node
                node = node[part]
            parent.pop(old_parts[-1])
        except KeyError:
            raise KeyError(f'Key {old_name} not found.')
        # 2. Add the new node.
        new_parts = new_name.split('.')
        self._recursively_add_leaf(self.tree, new_parts, new_name)


    def _recursively_add_leaf(self, node, parts, full_key):
        cur_part, rest_parts = parts[0], parts[1:]
        if len(rest_parts) == 0:
            assert cur_part not in node, f'Key {full_key} already exists.'
            node[cur_part] = full_key
        else:
            if cur_part not in node:
                node[cur_part] = {}
            self._recursively_add_leaf(node[cur_part], rest_parts, full_key)


    def _recursively_build_rich_tree(self, rich_node, dict_node, depth, max_depth:int=-1):
        if max_depth > 0 and depth >= max_depth:
            rich_node.add(f'... {len(dict_node)} more')
            return

        keys = sorted(dict_node.keys())
        for key in keys:
            next_dict_node = dict_node[key]
            next_rich_node = rich_node.add(key)
            if isinstance(next_dict_node, Dict):
                self._recursively_build_rich_tree(next_rich_node, next_dict_node, depth+1, max_depth)
