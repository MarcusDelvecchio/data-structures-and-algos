import random
class RandomizedSet:

    def __init__(self):
        self.random_set = {}
        self.idx_key_mapping = {}
        self.key_idx_mapping = {}

    def insert(self, val: int) -> bool:
        if not val: return True

        if val not in self.random_set:
            self.random_set[val] = True

            # update key mapping
            index = len(self.random_set.keys()) - 1
            self.idx_key_mapping[index] = val
            self.key_idx_mapping[val] = index
            return True
        return False

    def remove(self, val: int) -> bool:
        if not val: return False

        if val not in self.random_set:
            return False

        # remove the item from the set
        self.random_set.pop(val)

        print(self.idx_key_mapping)
        print(self.key_idx_mapping)

        index_of_key = self.key_idx_mapping[val]
        index_of_last_key = len(self.idx_key_mapping.keys()) - 1

        if index_of_key == index_of_last_key:
            self.key_idx_mapping.pop(val)
            self.idx_key_mapping.pop(index_of_key)
            return True

        print(index_of_key, index_of_last_key)

        # replace the current index of the item being removed with the item at the last index
        last_item = self.idx_key_mapping[index_of_last_key]
        self.idx_key_mapping[index_of_key] = last_item
        self.idx_key_mapping.pop(index_of_last_key)

        # replace the value at the index-value map 
        self.key_idx_mapping[self.idx_key_mapping[index_of_key]] = index_of_key
        self.key_idx_mapping.pop(val)

        # remove the entry for the item at the end as it was moved to the location of the item being removed
        # self.idx_key_mapping.pop(index_of_last_key)

        # remove the item being removed from the key-idx map
        # self.key_idx_mapping.pop(val)
        return True

    def getRandom(self) -> int:
        print(self.random_set.keys())
        if not len(self.random_set.keys()): return

        if len(self.random_set.keys()) == 1: 
            return self.idx_key_mapping[0]

        idx = random.randint(0, len(self.random_set.keys()) - 1)
        # print("idx_key_mapping:")
        # print(self.idx_key_mapping)
        # print(self.key_idx_mapping)
        # print(self.random_set)
        return self.idx_key_mapping[idx]


test = RandomizedSet()
test.remove(0)
test.remove(0)
test.insert(10)
# test.insert(20)
test.remove(10)
print(test.getRandom())
print(test.getRandom())
print(test.getRandom())
print(test.getRandom())
print(test.getRandom())
print(test.getRandom())
print(test.getRandom())
print(test.getRandom())