# Sample structure of a parameters object
# based on the structure of the sample NN structure in the same module
sample_parameters = {
    0 : {

    },

    1: {
        'W' : 'A 50 X 100 NP Array',
        'b' : 'A 50 X 1 NP Array'
    },

    2: {
        'W' : 'A 50 X 50 NP Array',
        'b' : 'A 50 X 1 NP Array',
        'A' : 'Added after forward Pass',
        'Forward Cache': {}
    },

    3: {
        'W' : 'A 50 X 50 NP Array',
        'b' : 'A 50 X 1 NP Array'
    },

    4: {
        'W' : 'A 1 X 50 NP Array',
        'b' : 'A 1 X 1 NP Array'
    }
}