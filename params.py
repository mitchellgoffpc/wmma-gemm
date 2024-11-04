PARAMS = {
    '01_simple_wmma': {
        'M_TILE': 16,
        'N_TILE': 16,
        'K_TILE': 16,
        'SHMEM_SIZE': 0,
        'BLOCK_SIZE': lambda params: params['WARP_SIZE'],
        'GRID_SIZE': lambda params: (params['M_GLOBAL'] * params['N_GLOBAL']) // (params['M_TILE'] * params['N_TILE']),
    },

    '02_shared_memory': {
        'M_TILE': 16,
        'N_TILE': 16,
        'K_TILE': 16,
        'TILE_ROWS_PER_BLOCK': 4,
        'TILE_COLS_PER_BLOCK': 4,
        'TILE_DEPTH_PER_CHUNK': 8,
        'CHUNK_PADDING': 8,
        'K_CHUNK': lambda params: params['TILE_DEPTH_PER_CHUNK'] * params['K_TILE'],
        'M_BLOCK': lambda params: params['M_TILE'] * params['TILE_ROWS_PER_BLOCK'],
        'N_BLOCK': lambda params: params['N_TILE'] * params['TILE_COLS_PER_BLOCK'],
        'SHMEM_SIZE': lambda params: 2 * (params['M_BLOCK'] * (params['K_CHUNK'] + params['CHUNK_PADDING']) + params['K_CHUNK'] * (params['N_BLOCK'] + params['CHUNK_PADDING'])),
        'BLOCK_SIZE': lambda params: params['WARP_SIZE'] * params['TILE_ROWS_PER_BLOCK'] * params['TILE_COLS_PER_BLOCK'],
        'GRID_SIZE': lambda params: (params['M_GLOBAL'] * params['N_GLOBAL']) // (params['M_BLOCK'] * params['N_BLOCK']),
    },

    '03_warptiling': {
        'M_TILE': 16,
        'N_TILE': 16,
        'K_TILE': 16,
        'TILE_ROWS_PER_BLOCK': 8,
        'TILE_COLS_PER_BLOCK': 8,
        'TILE_ROWS_PER_WARP': 4,
        'TILE_COLS_PER_WARP': 2,
        'WARP_ROWS_PER_BLOCK': lambda params: params['TILE_ROWS_PER_BLOCK'] // params['TILE_ROWS_PER_WARP'],
        'WARP_COLS_PER_BLOCK': lambda params: params['TILE_COLS_PER_BLOCK'] // params['TILE_COLS_PER_WARP'],
        'TILE_DEPTH_PER_CHUNK': 4,
        'CHUNK_PADDING': 8,
        'K_CHUNK': lambda params: params['TILE_DEPTH_PER_CHUNK'] * params['K_TILE'],
        'M_BLOCK': lambda params: params['M_TILE'] * params['TILE_ROWS_PER_BLOCK'],
        'N_BLOCK': lambda params: params['N_TILE'] * params['TILE_COLS_PER_BLOCK'],
        'SHMEM_SIZE': lambda params: 2 * (params['M_BLOCK'] * (params['K_CHUNK'] + params['CHUNK_PADDING']) + params['K_CHUNK'] * (params['N_BLOCK'] + params['CHUNK_PADDING'])),
        'BLOCK_SIZE': lambda params: params['WARP_SIZE'] * params['WARP_ROWS_PER_BLOCK'] * params['WARP_COLS_PER_BLOCK'],
        'GRID_SIZE': lambda params: (params['M_GLOBAL'] * params['N_GLOBAL']) // (params['M_BLOCK'] * params['N_BLOCK']),
    },

    '04_vectorized_load': {
        'M_TILE': 16,
        'N_TILE': 16,
        'K_TILE': 16,
        'TILE_ROWS_PER_BLOCK': 8,
        'TILE_COLS_PER_BLOCK': 8,
        'TILE_ROWS_PER_WARP': 4,
        'TILE_COLS_PER_WARP': 2,
        'WARP_ROWS_PER_BLOCK': lambda params: params['TILE_ROWS_PER_BLOCK'] // params['TILE_ROWS_PER_WARP'],
        'WARP_COLS_PER_BLOCK': lambda params: params['TILE_COLS_PER_BLOCK'] // params['TILE_COLS_PER_WARP'],
        'TILE_DEPTH_PER_CHUNK': 4,
        'CHUNK_PADDING': 8,
        'K_CHUNK': lambda params: params['TILE_DEPTH_PER_CHUNK'] * params['K_TILE'],
        'M_BLOCK': lambda params: params['M_TILE'] * params['TILE_ROWS_PER_BLOCK'],
        'N_BLOCK': lambda params: params['N_TILE'] * params['TILE_COLS_PER_BLOCK'],
        'SHMEM_SIZE': lambda params: 2 * (params['M_BLOCK'] * (params['K_CHUNK'] + params['CHUNK_PADDING']) + params['K_CHUNK'] * (params['N_BLOCK'] + params['CHUNK_PADDING'])),
        'BLOCK_SIZE': lambda params: params['WARP_SIZE'] * params['WARP_ROWS_PER_BLOCK'] * params['WARP_COLS_PER_BLOCK'],
        'GRID_SIZE': lambda params: (params['M_GLOBAL'] * params['N_GLOBAL']) // (params['M_BLOCK'] * params['N_BLOCK']),
    },

    '05_pipelined_load': {
        'M_TILE': 16,
        'N_TILE': 16,
        'K_TILE': 16,
        'TILE_ROWS_PER_BLOCK': 8,
        'TILE_COLS_PER_BLOCK': 8,
        'TILE_ROWS_PER_WARP': 4,
        'TILE_COLS_PER_WARP': 2,
        'WARP_ROWS_PER_BLOCK': lambda params: params['TILE_ROWS_PER_BLOCK'] // params['TILE_ROWS_PER_WARP'],
        'WARP_COLS_PER_BLOCK': lambda params: params['TILE_COLS_PER_BLOCK'] // params['TILE_COLS_PER_WARP'],
        'TILE_DEPTH_PER_CHUNK': 2,
        'CHUNK_PADDING': 8,
        'K_CHUNK': lambda params: params['TILE_DEPTH_PER_CHUNK'] * params['K_TILE'],
        'M_BLOCK': lambda params: params['M_TILE'] * params['TILE_ROWS_PER_BLOCK'],
        'N_BLOCK': lambda params: params['N_TILE'] * params['TILE_COLS_PER_BLOCK'],
        'SHMEM_SIZE': lambda params: 2 * 2 * (params['M_BLOCK'] * (params['K_CHUNK'] + params['CHUNK_PADDING']) + params['K_CHUNK'] * (params['N_BLOCK'] + params['CHUNK_PADDING'])),
        'BLOCK_SIZE': lambda params: params['WARP_SIZE'] * params['WARP_ROWS_PER_BLOCK'] * params['WARP_COLS_PER_BLOCK'],
        'GRID_SIZE': lambda params: (params['M_GLOBAL'] * params['N_GLOBAL']) // (params['M_BLOCK'] * params['N_BLOCK']),
    },
}

def get_params(kernel_name, global_params):
    params = PARAMS[kernel_name] | global_params
    while any(callable(x) for x in params.values()):
        n_updated = 0
        for k, v in params.items():
            if callable(v):
                try:
                    params[k] = v(params)
                    n_updated += 1
                except KeyError:
                    continue
        if n_updated == 0:
            raise ValueError("Could not resolve all parameters")
    return params
