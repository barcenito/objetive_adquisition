// Copyright (c) Acconeer AB, 2025
// All rights reserved
// This file is subject to the terms and conditions defined in the file
// 'LICENSES/license_acconeer.txt', (BSD 3-Clause License) which is part
// of this source code package.

#ifndef EXAMPLE_CARGO_H_
#define EXAMPLE_CARGO_H_

#include <stdbool.h>
#include <stdint.h>

#include "acc_config.h"
#include "acc_detector_distance.h"
#include "acc_detector_presence.h"
#include "acc_processing.h"
#include "acc_sensor.h"

/**
 * @brief Cargo Example App Presets
 */
typedef enum
{
	CARGO_PRESET_CONTAINER_10_FEET,
	CARGO_PRESET_CONTAINER_20_FEET,
	CARGO_PRESET_CONTAINER_40_FEET,
	CARGO_PRESET_NO_LENS,
} cargo_preset_t;

/**
 * @brief Container size
 */
typedef enum
{
	CARGO_CONTAINER_SIZE_10_FT,
	CARGO_CONTAINER_SIZE_20_FT,
	CARGO_CONTAINER_SIZE_40_FT,
} cargo_container_size_t;

/**
 * @brief Configuration struct for cargo
 */
typedef struct
{
	uint32_t magic;

	cargo_container_size_t container_size;

	bool activate_utilization_level;

	struct
	{
		/**
		 * Signal quality (dB).
		 * High quality equals higher HWAAS and better SNR but increases power consumption.
		 */
		float signal_quality;

		/**
		 * Sensitivity of threshold.
		 *
		 * High sensitivity equals low detection threshold, low sensitivity equals high detection
		 * threshold.
		 */
		float threshold_sensitivity;
	} utilization;

	bool activate_presence;

	struct
	{
		/**
		 * Sets the presence detector update rate (Hz).
		 */
		float update_rate;

		/**
		 * Number of sweeps per frame.
		 */
		uint16_t sweeps_per_frame;

		/**
		 * Signal quality (dB).
		 * High quality equals higher HWAAS and better SNR but increases power consumption.
		 */
		float signal_quality;

		/**
		 * Detection threshold for the inter-frame presence detection
		 */
		float inter_detection_threshold;

		/**
		 * Detection threshold for the intra-frame presence detection.
		 */
		float intra_detection_threshold;

		/**
		 * Whether the Presence's frame rate should be app driven.
		 * See @ref acc_detector_presence_config_frame_rate_app_driven_set
		 */
		bool rate_is_app_driven;
	} presence;
} cargo_config_t;

/**
 * @brief Result type
 */
typedef struct
{
	/** If true, the sensor needs to be recalibrated & the cargo calibration needs to be updated */
	bool calibration_needed;

	/** Whether fields distance, level_m & level_percent are valid */
	bool utilization_valid;

	/** The distance in meters to the measured object */
	float distance;

	/** The container fill level in meters.
	 * Ranges from 0m to the full container size (10, 20 or 40 ft)
	 */
	float level_m;

	/** The container fill level in percent.
	 * Ranges from 0% to 100%
	 */
	float level_percent;

	/** Whether fields presence_detected, inter_presence_score & intra_presence_score are valid */
	bool presence_valid;

	/** Whether presence was detected.  */
	bool presence_detected;

	/** See @ref acc_detector_presence_result_t */
	float inter_presence_score;

	/** See @ref acc_detector_presence_result_t */
	float intra_presence_score;
} cargo_result_t;

/**
 * @brief Handle type (forward declared)
 */
struct cargo_handle;

typedef struct cargo_handle cargo_handle_t;

/**
 * @brief Cargo mode enumeration
 */
typedef enum
{
	CARGO_MODE_NONE = 0,
	CARGO_MODE_UTILIZATION,
	CARGO_MODE_PRESENCE,
} cargo_mode_t;

/**
 * @brief Initialize cargo config struct with a preset
 *
 * @param[out] cargo_config The cargo config to initialize
 * @param[in] preset A @ref cargo_preset_t to initialize the config to
 */
void cargo_config_initialize(cargo_config_t *cargo_config, cargo_preset_t preset);

/**
 * @brief Log cargo config to debug uart
 *
 * @param[in] cargo_config The cargo config to log
 */
void cargo_config_log(const cargo_config_t *cargo_config);

/**
 * @brief Create a cargo handle
 *
 * @param[in] cargo_config Cargo configuration
 * @return A cargo handle, or NULL if creation failed
 */
cargo_handle_t *cargo_handle_create(const cargo_config_t *cargo_config);

/**
 * @brief Destroy a cargo handle
 *
 * @param[in] handle The cargo handle to destroy
 */
void cargo_handle_destroy(cargo_handle_t *handle);

/**
 * @brief Get the required buffer sensor data buffer size
 *
 * @param[in] handle The cargo handle
 * @param[out] buffer_size The required buffer size
 * @return True if successful, false otherwise
 */
bool cargo_get_buffer_size(const cargo_handle_t *handle, uint32_t *buffer_size);

/**
 * @brief Get the required buffer for static distance detector calibration
 *
 * @param[in] handle The cargo handle
 * @param[out] distance_cal_result_static_size The required buffer size.
 *                                             Will be 0U if activate_utilization_level is false.
 * @return True if successful, false otherwise
 */
bool cargo_get_distance_cal_result_static_size(const cargo_handle_t *handle, uint32_t *distance_cal_result_static_size);

/**
 * @brief Calibrate Cargo
 *
 * Calibrating Cargo's internal Detector is needed to ensure reliable Detector results.
 *
 * See the calibration sections in the Distance Detector User Guide for more information.
 *
 * @param[in] sensor The sensor instance to use for calibration
 * @param[in] handle The Cargo handle
 * @param[in] sensor_cal_result Sensor calibration result
 * @param[in] buffer Working memory buffer needed by function
 * @param[in] buffer_size The size of buffer. Needs to be at least
 *            the result of @ref cargo_get_buffer_size
 * @param[out] distance_cal_result_static Static result of calibration
 * @param[in] distance_cal_result_static_size The size of distance_cal_result_static.
 *            Needs to be at least the result of @ref cargo_get_distance_cal_result_static_size
 * @param[out] distance_cal_result_dynamic Dynamic result of calibration
 * @param[out] calibration_complete Will be set to true when the calibration is complete.
 *             If false; at least one more call to this function is needed.
 *             Note that it's necessary to wait for interrupt between calls.
 * @return True if successful, false otherwise
 */
bool cargo_calibrate(cargo_handle_t                    *handle,
                     acc_sensor_t                      *sensor,
                     const acc_cal_result_t            *sensor_cal_result,
                     void                              *buffer,
                     uint32_t                           buffer_size,
                     uint8_t                           *distance_cal_result_static,
                     uint32_t                           distance_cal_result_static_size,
                     acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic,
                     bool                              *calibration_complete);

/**
 * @brief Update Cargo calibration
 *
 * Updates Cargo's internal calibration.
 *
 * If calibration_needed has been set to true, you should do the following:
 *  1. Recalibrate sensor (see @ref acc_sensor_calibrate)
 *  2. Call this function in a loop until calibration_complete=true (and while true is returned)
 *  3. Redo the Utilization- or Presence measurement
 *
 * @param[in] sensor The sensor instance to use for calibration
 * @param[in] handle The Cargo handle
 * @param[in] sensor_cal_result Sensor calibration result
 * @param[in] buffer Working memory buffer needed by function
 * @param[in] buffer_size The size of buffer. Needs to be at least
 *            the result of @ref cargo_get_buffer_size
 * @param[out] distance_cal_result_dynamic Dynamic result of calibration
 * @param[out] calibration_complete Will be set to true when the calibration is complete.
 *             If false; at least one more call to this function is needed.
 *             Note that it's necessary to wait for interrupt between calls.
 * @return True if successful, false otherwise
 */
bool cargo_update_calibration(cargo_handle_t                    *handle,
                              acc_sensor_t                      *sensor,
                              const acc_cal_result_t            *sensor_cal_result,
                              void                              *buffer,
                              uint32_t                           buffer_size,
                              acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic,
                              bool                              *calibration_complete);

/**
 * @brief Prepare Cargo utilization for measurements
 *
 * @param[in, out] handle The Cargo handle
 * @param[in] sensor The sensor instance
 * @param[in] sensor_cal_result The sensor calibration result
 * @param[in] buffer Memory used by the detector. Should be at least buffer_size bytes
 * @param[in] buffer_size The buffer size received by @ref acc_detector_distance_get_sizes
 * @return True if successful, false otherwise
 */
bool cargo_prepare_utilization(cargo_handle_t         *handle,
                               acc_sensor_t           *sensor,
                               const acc_cal_result_t *sensor_cal_result,
                               void                   *buffer,
                               uint32_t                buffer_size);

/**
 * @brief Prepare Cargo presence for measurements
 *
 * @param[in, out] handle The Cargo handle
 * @param[in] sensor The sensor instance
 * @param[in] sensor_cal_result The sensor calibration result
 * @param[in] buffer Memory used by the detector. Should be at least buffer_size bytes
 * @param[in] buffer_size The buffer size received by @ref acc_detector_distance_get_sizes
 * @return True if successful, false otherwise
 */
bool cargo_prepare_presence(cargo_handle_t         *handle,
                            acc_sensor_t           *sensor,
                            const acc_cal_result_t *sensor_cal_result,
                            void                   *buffer,
                            uint32_t                buffer_size);

/**
 * @brief Process Sparse IQ data into a Cargo result
 *
 * @param[in] handle The Cargo handle
 * @param[in] distance_cal_result_static The result from @ref cargo_calibrate
 * @param[in] distance_cal_result_dynamic The result from @ref cargo_calibrate
 * @param[in] buffer A reference to the buffer (populated by @ref acc_sensor_read) containing the
 *                   data to be processed.
 * @param[out] cargo_result Cargo result
 * @param[out] result_available True if the cargo_result is valid
 * @param[out] nullable_distance_result Pointer to auxilliary @ref acc_detector_distance_result_t
 *                                      Can be NULL.
 * @return True if successful, false otherwise
 */
bool cargo_process_utilization(cargo_handle_t                    *handle,
                               uint8_t                           *distance_cal_result_static,
                               acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic,
                               void                              *buffer,
                               cargo_result_t                    *cargo_result,
                               bool                              *result_available,
                               acc_detector_distance_result_t    *nullable_distance_result);

/**
 * @brief Process Sparse IQ data into a Cargo result
 *
 * @param[in] handle The Cargo handle
 * @param[in] buffer A reference to the buffer (populated by @ref acc_sensor_read) containing the
 *                   data to be processed.
 * @param[out] cargo_result Cargo result
 * @param[out] nullable_presence_result Pointer to auxilliary @ref acc_detector_presence_result_t
 *                                      Can be NULL.
 * @return True if successful, false otherwise
 */
bool cargo_process_presence(cargo_handle_t                 *handle,
                            void                           *buffer,
                            cargo_result_t                 *cargo_result,
                            acc_detector_presence_result_t *nullable_presence_result);

/**
 * @brief Get current mode of the cargo handle
 *
 * Really only useful when doing both Presence- & Utilization level measurements.
 *
 * @param[in] handle The cargo handle
 * @return The current cargo mode. See @ref cargo_mode_t
 */
cargo_mode_t cargo_current_mode_get(const cargo_handle_t *handle);

/**
 * @brief Get presence detector metadata
 *
 * @param[in] handle The cargo handle
 * @param[out] presence_metadata The Presence Detector metadata
 * @return True if handle is valid & configured with presence activated, false otherwise
 */
bool cargo_presence_metadata_get(cargo_handle_t *handle, acc_detector_presence_metadata_t *presence_metadata);

#endif
