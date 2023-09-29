import tensorflow as tf
from Note.nn.layer.image_preprocessing.transform import transform
from Note.nn.layer.image_preprocessing.get_translation_matrix import get_translation_matrix


H_AXIS = -3
W_AXIS = -2


class random_translation:
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=7,
        fill_value=0.0,
    ):
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                f"lower bound, got {height_factor}"
            )
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` argument must have values between [-1, 1]. "
                f"Received: height_factor={height_factor}"
            )

        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                f"lower bound, got {width_factor}"
            )
        if abs(self.width_lower) > 1.0 or abs(self.width_upper) > 1.0:
            raise ValueError(
                "`width_factor` must have values between [-1, 1], "
                f"got {width_factor}"
            )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.random_generator = tf.random.Generator.from_seed(seed)

    def output(self, data, train_flag=True):
        def random_translated_inputs(data):
            """Translated inputs with random ops."""
            # The transform op only accepts rank 4 inputs,
            # so if we have an unbatched image,
            # we need to temporarily expand dims to a batch.
            original_shape = data.shape
            unbatched = data.shape.rank == 3
            if unbatched:
                data = tf.expand_dims(data, 0)

            inputs_shape = tf.shape(data)
            batch_size = inputs_shape[0]
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            height_translate = self.random_generator.uniform(
                shape=[batch_size, 1],
                minval=self.height_lower,
                maxval=self.height_upper,
                dtype=tf.float32,
            )
            height_translate = height_translate * img_hd
            width_translate = self.random_generator.uniform(
                shape=[batch_size, 1],
                minval=self.width_lower,
                maxval=self.width_upper,
                dtype=tf.float32,
            )
            width_translate = width_translate * img_wd
            translations = tf.cast(
                tf.concat([width_translate, height_translate], axis=1),
                dtype=tf.float32,
            )
            output = transform(
                data,
                get_translation_matrix(translations),
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output

        if train_flag:
            return random_translated_inputs(data)
        else:
            return data