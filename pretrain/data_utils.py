import torch

def apply_with_random_selector(x, func, num_cases):
    sel = torch.rand([],dtype=torch.int32)

def flip_if_vertical(image):
    height = image.shape[0]
    width = image.shape[1]
    if height >= (4*width/3.0):
        image = torch.nn.functional.pad(torch.rot90(image), (0,0,4,4,0,0), mode="constant", value=0.5)
    return image


def resize_and_pad(image, desired_output_size,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True,
                   do_flip_if_vertical=True,
                   resize_method="random"):

    if do_flip_if_vertical:# padding 
        image = flip_if_vertical(image)
    desired_height = torch.tensor(desired_output_size[0], dtype=float) #192
    desired_width = torch.tensor(desired_output_size[1], dtype=float) #320

    height = torch.tensor(image.shape[1], dtype=float) #288 -> 192
    width = torch.tensor(image.shape[2], dtype=float) #512 -> 320

    if do_random_scale:
        random_scale_factor = torch.rand([])*(random_scale_max-random_scale_min)+ random_scale_min
        if not shrink_both_sides:
            rsf_max = torch.maximum(desired_width/width, desired_height/height)
            random_scale_factor = torch.minimum(rsf_max, random_scale_factor)
            
        scaled_y = (random_scale_factor * desired_height).int() #165
        scaled_x = (random_scale_factor * desired_width).int() #276
        
        image_scale_y = scaled_y.float() / height
        image_scale_x = scaled_x.float() / width
        image_scale = torch.minimum(image_scale_x, image_scale_y) #0.5391

        image_scale = torch.maximum(image_scale, 64.0 / torch.minimum(height, width))

        scaled_height = (height * image_scale).int() #64
        scaled_width = (width * image_scale).int() #113

        offset_y = (scaled_height - desired_height).float()
        offset_x = (scaled_width - desired_width).float()
        offset_y = (torch.maximum(torch.zeros(1), offset_y.clone().detach()) * torch.rand([])).int().item()
        offset_x = (torch.maximum(torch.zeros(1), offset_x.clone().detach()) * torch.rand([])).int().item()
        
    else:
        image_scale_y = desired_height / height
        image_scale_x = desired_width / width
        image_scale = torch.minimum(image_scale_x, image_scale_y)
        scaled_height = (height * image_scale).int()
        scaled_width = (width * image_scale).int()
        offset_y = torch.zeros(1)
        offset_x = torch.zeros(1)

    # resize and crop
    if resize_method =="random" and do_random_scale : # tensorflow에서 지원하는 resize 방법 일부가 torch에서 지원 안됨
        image = transforms.Resize((scaled_height, scaled_width), antialias=True)(image) # 0~255 사이값으로 텐서값 변경, 
        image /= torch.max(image)

    image = torch.clamp(image, 0.0, 1.0)
    image = image[:, offset_y:offset_y+desired_output_size[0],
                  offset_x:offset_x+desired_output_size[1]]

    if image.shape[1] != desired_output_size[0]:
        n_pad = desired_output_size[0] - image.shape[1]
        image = torch.cat((image,torch.zeros(image.shape[0], n_pad, image.shape[2])), dim=1)
        
    if image.shape[2] != desired_output_size[1]:
        n_pad = desired_output_size[1]-image.shape[2]
        image = torch.cat((image, torch.zeros(image.shape[0],image.shape[1], n_pad)), dim=2)

    effective_height = torch.minimum(scaled_height, desired_height)
    effective_width = torch.minimum(scaled_width, desired_width)
    image_info = torch.stack([effective_height.float() / desired_height, 
                              effective_width.float() / desired_width,
                              1.0/image_scale,
                              height, 
                              width, 
                              offset_y / height,
                              offset_x / width])
    return image, image_info

def cumulative_maximum_int(x):
    assert x.dtype ==  torch.int32
    N = list(x.shape)[0]
    x_tile = torch.tile(x[None], (N, 1))
    arange_x = torch.arange(0, N)
    valid = torch.greater_equal(arange_x[:, None], arange_x[None])
    x_tile = torch.where(valid, x_tile, torch.full([N, N], -2147483648).int())
    return torch.max(x_tile, -1)[0]


def uniform_random_select(n, num_samples, sort_idx=True):
    if isinstance(num_samples, int) and isinstance(n, int):
        assert num_samples <= n
    logits = torch.rand([n])
    idx = torch.argsort(logits)[:num_samples]
    if sort_idx:
        idx = torch.sort(idx)[0]
        
    return idx


def random_categorical_without_replacement(logits, num_samples):
    z = - torch.log(-torch.log(torch.rand(logits.shape)))
    _, indices = torch.topk(logits+z, num_samples)
    return indices.int()

def pad_tokens_to_fixed_size(tokens, padded_seq_len):
    missing_len = torch.maximum(torch.tensor(padded_seq_len-tokens.shape[0]), torch.tensor(0))
    dummy_row = torch.tensor([0, -1, -1], dtype=torch.int32)
    tokens = torch.concat([tokens, torch.tile(dummy_row[None], [missing_len, 1])], 0)[:padded_seq_len]
    tokens = tokens.reshape(padded_seq_len, 3)
    return tokens



class RaggedTensor:
    def __init__(self, ragged_tensor, nrows=None, row_len=None,value_rowids=None, dtype=None):
        # 무조건 2차원.
        self._nrows = torch.tensor(len(ragged_tensor)) if nrows is None else nrows
        self.dtype = dtype if dtype is None else type(ragged_tensor[0][0])
        self._value = [torch.tensor(t) for t in ragged_tensor]
        self._row_len = torch.tensor(row_len) if row_len is not None else None
        self._value_rowids_ = torch.tensor(value_rowids) if value_rowids is not None else None
        
        if self._row_len is None:
            self._row_len = [[i]*len(r_t)  for i, r_t in enumerate(ragged_tensor)]
            self._row_len = self.flatten_(self._row_len)

    def __getitem__(self, idx):
        return self._value[idx]

    def __getvalues__(self):
        return self._value
    

    def values(self):
        return torch.cat(self._value, dim=-1)
    
    def row_lengths(self):
        return self._row_len
    
        
    def bounding_shape(self):
        return [self._nrows, None]
    
    def nrows(self):
        return self._nrows
    
        
    def value_rowids(self):
        return self._value_rowids_
        
    @classmethod
    def from_value_rowids(cls, values, value_rowids, nrows):
        result = [[values[i] for i in range(len(values)) if value_rowids[i] == row] for row in range(nrows)]
        row_len = [len(i) for i in result]
        return RaggedTensor(result, nrows=nrows, row_len=row_len, value_rowids=value_rowids)
        

    @classmethod
    def from_row_lengths(cls, values, row_lengths):
        values = list(torch.cat(cls.flatten_(values), -1).numpy())
        result = [[values.pop(0)  for i in range(length)] for length in row_lengths]
        nrows = len(result)
        value_rowids = torch.tensor(cls.flatten_([[i]*row for i, row in enumerate(row_lengths)]))
        return RaggedTensor(result, nrows=nrows, row_len=row_lengths, value_rowids=value_rowids)
     
    @classmethod
    def flatten_(cls, data):
        output = []
        for i in data:
            if isinstance(i, list):
                output += flatten(i)
            else:
                output += [i]
        return output

        
def ragged_from_rowids(values, value_rowids, nrows):
    result = [[values[i] for i in range(len(values)) if value_rowids[i] == row] for row in range(nrows)]
    return result

def ragged_from_rowlen(values, row_lengths):
    values = flatten(values)
    result = [[values.pop(0) for i in range(length)] for length in row_lengths]
    return result

def flatten(data):
    output = []
    for i in data:
        if isinstance(i, list):
            output += flatten(i)
        else:
            output += [i]
    return output


def sample_bernoulli(p_a):
    if isinstance(p_a, float):
        if p_a == 0.0:
            print("sample_bernoulli p_a == 0.0: retrun False")
            return torch.tensor([False])
        elif p_a == 1.0:
            print("sample_bernoulli p_a == 0.0: return True")
            return torch.tensor([True])
    is_a = torch.distributions.categorical.Categorical(torch.log(torch.tensor([[1.0-p_a, p_a]])))
    is_a = is_a.sample().bool()
    return is_a 


def sample_bernoullis(p_a, N=1):
    if isinstance(p_a, float):
        if p_a ==0.0:
            print("sample_bernoulli p_a == 0.0:return False")
            return torch.tensor(False for i in range(N))
        elif p_a == 1.0:
            print("sample_bernoulli p_a ==0.0:return True")
            return torch.tensor([True for i in range(N)])
    sampler = torch.distributions.categorical.Categorical(torch.log(torch.tensor([[1.0-p_a, 0.5]])))
    is_a = torch.stack([sampler.sample() for s in range(N)]).squeeze()
    return is_a 

def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))

    