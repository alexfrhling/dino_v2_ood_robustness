# Task: set of category names -> group by semantic similarity
# Steps:
# 1) Tree of wnid-nodes structured through is-a relationships 
# 2) Mapping: category-name -> wnid 
# 3) Given a category of intersection -> find all neighbouring categories in tree 
# 4) Check if one of the neighbours is also in intersection 
#  
# reads words.txt: (wnid, [list of synonyms])

def map_wnid_to_classnames():
    map_wnid_to_classname = dict()

    with open('../resources/classnames.txt', 'r') as classnames_file: 
        for line in classnames_file.readlines():
            splitted_line = line.split(' ')
            wnid = splitted_line[0]
            classname = ''
            for i in range(1, len(splitted_line)):
                classname = classname + ' ' + splitted_line[i]

            classname = classname.rstrip('\n')
            map_wnid_to_classname[wnid] = classname  

    return map_wnid_to_classname

def map_wnid_to_word():
    map_wnid_to_word = dict()

    with open('../resources/words.txt', 'r') as words_file: 
        for line in words_file.readlines():
            splitted_line = line.split('\t')
            # extract wnid of line
            wnid = splitted_line[0]
            # only map to the first word related to wnid in right column
            word = splitted_line[1].split(',')
            word = word[0]
            word = word.rstrip('\n')

            map_wnid_to_word[wnid] = word  

    return map_wnid_to_word


class ImagenetSemanticInfo():
     
     def __init__(self):
          self.wnid_to_classname = map_wnid_to_classnames() 
          self.wnid_to_word = map_wnid_to_word()
          self.semantic_tree = ImagenetSemanticTree()
     

class WnidNode():

    def __init__(self, wnid: str):
        self.wnid = wnid 
        self.children = [] 
        self.parent = None
        self.is_on_path = False

    def add_child(self, new_child): 
        self.children.append(new_child)

    def add_parent(self, parent_node):
        self.parent = parent_node


class ImagenetSemanticTree():

    def __init__(self):
        self.all_nodes = set()
        self.root = None 
        self.tree_depth = None
        self.tree = dict()
        self.build_tree() 


    def check_tree(self) -> bool:
        if len(self.all_nodes) == 0:
            return False 
        
        with open(f'../resources/wordnet.is_a.txt') as is_a_file: 
            n = 0
            for line in is_a_file.readlines():
                parent_wnid, child_wnid = line.split(' ')
                child_wnid = child_wnid.rstrip('\n')

                parent_node = self.tree[parent_wnid]
                children_wnids = [child.wnid for child in parent_node.children]

                if not child_wnid in children_wnids:
                    return False 
                
                n = n + 1 
            
            print(f'Correct classified is-a relations: {n}')

        return True
                

    def build_tree(self):

        with open(f'../resources/wordnet.is_a.txt') as is_a_file: 
            for line in is_a_file.readlines():
                parent_wnid, child_wnid = line.split(' ')
                child_wnid = child_wnid.rstrip('\n')

                if child_wnid == 'n02066245':
                    print('Stop node discovered')
                    
                if parent_wnid in self.all_nodes: 
                    parent_node = self.tree[parent_wnid]

                    if child_wnid in self.all_nodes:
                        child_node = self.tree[child_wnid]

                        child_node.add_parent(parent_node)
                        parent_node.add_child(child_node)
                    else:
                        child_node = WnidNode(child_wnid)
                        self.all_nodes.add(child_wnid)
                        self.tree[child_wnid] = child_node 

                        child_node.add_parent(parent_node)
                        parent_node.add_child(child_node)
                        
                else:
                    parent_node = WnidNode(parent_wnid)
                    self.all_nodes.add(parent_wnid)
                    self.tree[parent_wnid] = parent_node 
                    # in case entity-node is created, mark it as root of Imagenet-Tree
                    if parent_wnid == 'n00001740':
                        self.root = parent_node

                    if child_wnid in self.all_nodes:
                        child_node = self.tree[child_wnid]

                        child_node.add_parent(parent_node)
                        parent_node.add_child(child_node)
                    else:
                        child_node = WnidNode(child_wnid)
                        self.all_nodes.add(child_wnid)
                        self.tree[child_wnid] = child_node

                        child_node.add_parent(parent_node)
                        parent_node.add_child(child_node)

    def get_node_by_wnid(self, wnid: str):

        return self.tree[wnid]
    
    def get_depth(self): 
        if self.tree_depth != None: 
            return self.tree_depth 
        else: 
            max_depth = 0 

            init_child_nodes = self.root.children 
            child_nodes = [(init_child, 1) for init_child in init_child_nodes]

            while len(child_nodes) > 0: 
                current_child, current_depth = child_nodes.pop()
                next_children = current_child.children 

                if len(next_children) > 0:
                    child_nodes.extend([(next_child, current_depth + 1) for next_child in next_children])
                else:
                    if current_depth > max_depth:
                        max_depth = current_depth 

            self.tree_depth = max_depth 
            return self.tree_depth

class ImagenetSemanticSubtree():

    def __init__(self, imagenet_tree: ImagenetSemanticTree, ood_wnid: str, id_wnids: list[str], map_to_word: dict):
        self.imagenet_tree = imagenet_tree 
        self.root = None 
        self.ood_wnid = ood_wnid 
        self.id_wnids = id_wnids
        self.map_to_word = map_to_word

    def create_subtree(self, subtree_root: WnidNode): 
        self.root = subtree_root 

    def _help_build_trace(self, next_level_nodes: list[WnidNode]) -> str: 

        next_level_trace = '('
        for node in next_level_nodes: 
            if node.wnid != self.ood_wnid and not node.wnid in self.id_wnids:
                next_level_trace = next_level_trace + ' ' + self.map_to_word[node.wnid]
            else: 
                next_level_trace = next_level_trace + ' ' + '\033[1m' + self.map_to_word[node.wnid] + '\033[0m'

            if node.is_on_path:
                next_level_trace = next_level_trace + ' ' + self._help_build_trace(node.children)
            else:
                next_level_trace = next_level_trace + ' (...) '
                
        next_level_trace = next_level_trace + ')'

        return next_level_trace 
    
    def parse_tree(self) -> str: 

        ## Trace represented as: (0 [root] (01 (011 012)) (02) (03)) 

        trace = self._help_build_trace([self.root])
        return trace

    def propagate_paths(self, init_nodes: list[WnidNode], visited_nodes: set[WnidNode]) -> bool:

        nodes_on_path = init_nodes
        root_reached = False 

        for node in visited_nodes: 
            node.is_on_path = True 

        # while len(nodes_on_path) > 0: 
        #     node_to_be_set = nodes_on_path.pop() 

        #     if (node_to_be_set.wnid == 'n01861778') and (self.root.wnid == 'n01861778'):
        #         print('Root-node of first subtree was reached')

        #     if node_to_be_set.parent.wnid == 'n00001740':
        #         print(f'Child of enitity is: {node_to_be_set.wnid}')

        #     if node_to_be_set.wnid == 'n00001740':
        #         root_reached = True 

        #     if node_to_be_set == None: 
        #         print('In propagate_paths: None retrived as next node')
        #         return False 
            
        #     node_to_be_set.is_on_path = True 

        #     if not node_to_be_set == self.root and not node_to_be_set.wnid == 'n00001740': 
        #         if node_to_be_set.parent == None:
        #             print(f'Child has None as parent: {node_to_be_set.wnid}, Root is: {self.root.wnid}')
        #         nodes_on_path.append(node_to_be_set.parent)

        # return root_reached

    def clear_paths(self, visited_nodes):

        for node in visited_nodes:
            node.is_on_path = False 
            
        # open_nodes = [self.root]
        # while len(open_nodes) > 0:
        #     node_to_be_cleared = open_nodes.pop()
        #     node_to_be_cleared.is_on_path = False 
        #     open_nodes.extend(node_to_be_cleared.children)





    

