import tensorflow as tf


def get_translation_matrix(translations):
    """Returns projective transform(s) for the given translation(s).

    Args:
        translations: A matrix of 2-element lists representing `[dx, dy]`
            to translate for each image (for a batch of images).

    Returns:
        A tensor of shape `(num_images, 8)` projective transforms
            which can be given to `transform`.
    """
    num_translations = tf.shape(translations)[0]
    # The translation matrix looks like:
    #     [[1 0 -dx]
    #      [0 1 -dy]
    #      [0 0 1]]
    # where the last entry is implicit.
    # Translation matrices are always float32.
    return tf.concat(
        values=[
            tf.ones((num_translations, 1), tf.float32),
            tf.zeros((num_translations, 1), tf.float32),
            -translations[:, 0, None],
            tf.zeros((num_translations, 1), tf.float32),
            tf.ones((num_translations, 1), tf.float32),
            -translations[:, 1, None],
            tf.zeros((num_translations, 2), tf.float32),
        ],
        axis=1,
    )