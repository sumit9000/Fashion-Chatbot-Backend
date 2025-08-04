<?php
/**
 * Fashion Chatbot API with Advanced Image Analysis
 * Supports both text-only conversations and fashion image analysis
 * Uses OpenAI GPT-4 Vision for image understanding
 */

// Set proper headers for API responses
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With');

// Increase memory and execution limits for image processing
ini_set('memory_limit', '256M');
ini_set('max_execution_time', 120);

/**
 * Load environment variables from .env file
 * @param string $path Path to .env file
 * @return bool Success status
 */
function loadEnvironmentVariables($path) {
    if (!file_exists($path)) {
        error_log("Environment file not found: $path");
        return false;
    }
    
    try {
        $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        foreach ($lines as $line) {
            $line = trim($line);
            
            // Skip comments and empty lines
            if (empty($line) || strpos($line, '#') === 0) {
                continue;
            }
            
            // Parse key=value pairs
            if (strpos($line, '=') !== false) {
                list($key, $value) = explode('=', $line, 2);
                $_ENV[trim($key)] = trim($value);
            }
        }
        return true;
    } catch (Exception $e) {
        error_log("Error loading environment variables: " . $e->getMessage());
        return false;
    }
}

/**
 * Validate uploaded image file
 * @param array $file $_FILES array element
 * @return array Validation result
 */
function validateImage($file) {
    // Check if file was uploaded
    if (!isset($file) || $file['error'] !== UPLOAD_ERR_OK) {
        return [
            'valid' => false,
            'error' => 'No valid image file uploaded'
        ];
    }
    
    // Validate file type
    $allowedTypes = [
        'image/jpeg' => '.jpg',
        'image/jpg' => '.jpg', 
        'image/png' => '.png',
        'image/gif' => '.gif',
        'image/webp' => '.webp'
    ];
    
    if (!array_key_exists($file['type'], $allowedTypes)) {
        return [
            'valid' => false,
            'error' => 'Invalid file type. Please upload JPG, PNG, GIF, or WebP images only.'
        ];
    }
    
    // Validate file size (max 10MB)
    $maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if ($file['size'] > $maxSize) {
        return [
            'valid' => false,
            'error' => 'Image too large. Please upload an image smaller than 10MB.'
        ];
    }
    
    // Additional security check - verify it's actually an image
    $imageInfo = getimagesize($file['tmp_name']);
    if ($imageInfo === false) {
        return [
            'valid' => false,
            'error' => 'Invalid image file. Please upload a valid image.'
        ];
    }
    
    return [
        'valid' => true,
        'type' => $file['type'],
        'size' => $file['size'],
        'width' => $imageInfo[0],
        'height' => $imageInfo[1]
    ];
}

/**
 * Convert image to base64 with optimization
 * @param string $tmpPath Temporary file path
 * @param string $mimeType MIME type of image
 * @return string Base64 encoded image
 */
function processImageForAPI($tmpPath, $mimeType) {
    // Read and encode image
    $imageData = file_get_contents($tmpPath);
    if ($imageData === false) {
        throw new Exception('Failed to read image file');
    }
    
    // For very large images, consider resizing to reduce API costs
    $imageSize = strlen($imageData);
    if ($imageSize > 2 * 1024 * 1024) { // 2MB
        // You could add image compression logic here if needed
        error_log("Large image uploaded: " . ($imageSize / 1024 / 1024) . "MB");
    }
    
    return base64_encode($imageData);
}

/**
 * Call OpenAI API for text-only fashion advice
 * @param string $message User message
 * @param string $apiKey OpenAI API key
 * @return array API response
 */
function getFashionAdvice($message, $apiKey) {
    $systemPrompt = "You are an expert fashion stylist and personal shopping assistant with years of experience in the fashion industry. You provide:

1. **Specific and actionable fashion advice**
2. **Trend-aware recommendations**
3. **Occasion-appropriate styling**
4. **Color coordination guidance**
5. **Body type and personal style considerations**
6. **Budget-friendly alternatives**
7. **Seasonal appropriateness**

Be enthusiastic, supportive, and detailed in your responses. Use fashion terminology appropriately and provide practical tips that users can implement immediately.";

    $requestData = [
        'model' => 'gpt-4o',
        'messages' => [
            [
                'role' => 'system',
                'content' => $systemPrompt
            ],
            [
                'role' => 'user',
                'content' => $message
            ]
        ],
        'max_tokens' => 1200,
        'temperature' => 0.7,
        'presence_penalty' => 0.1,
        'frequency_penalty' => 0.1
    ];
    
    return makeOpenAIRequest($requestData, $apiKey);
}

/**
 * Analyze fashion image using OpenAI Vision API
 * @param string $message User message/question
 * @param string $imageBase64 Base64 encoded image
 * @param string $mimeType Image MIME type
 * @param string $apiKey OpenAI API key
 * @return array API response
 */
function analyzeFashionImage($message, $imageBase64, $mimeType, $apiKey) {
    $systemPrompt = "You are a professional fashion stylist with expertise in:

- **Style Analysis**: Identifying clothing items, colors, patterns, and overall aesthetic
- **Fashion Critique**: Providing constructive feedback on fit, color coordination, and styling
- **Trend Knowledge**: Understanding current fashion trends and timeless style principles
- **Occasion Styling**: Recommending appropriate modifications for different settings
- **Personal Shopping**: Suggesting complementary pieces and accessories
- **Color Theory**: Understanding color palettes and what works with different skin tones

When analyzing fashion images:
1. Describe what you see in detail
2. Identify the style category and aesthetic
3. Comment on fit, colors, and coordination
4. Suggest improvements or alternatives
5. Recommend accessories or styling changes
6. Consider the occasion and appropriateness
7. Provide actionable next steps

Be specific, encouraging, and professional in your analysis.";

    // Default message if user doesn't provide one
    $userPrompt = empty(trim($message)) ? 
        "Please analyze this fashion image and provide detailed styling advice, including what works well and suggestions for improvement." : 
        $message;

    $requestData = [
        'model' => 'gpt-4o', // GPT-4 with vision capabilities
        'messages' => [
            [
                'role' => 'system',
                'content' => $systemPrompt
            ],
            [
                'role' => 'user',
                'content' => [
                    [
                        'type' => 'text',
                        'text' => $userPrompt
                    ],
                    [
                        'type' => 'image_url',
                        'image_url' => [
                            'url' => "data:$mimeType;base64,$imageBase64",
                            'detail' => 'high' // Use high detail for better analysis
                        ]
                    ]
                ]
            ]
        ],
        'max_tokens' => 1500,
        'temperature' => 0.6,
        'presence_penalty' => 0.1
    ];
    
    return makeOpenAIRequest($requestData, $apiKey);
}

/**
 * Make HTTP request to OpenAI API
 * @param array $data Request payload
 * @param string $apiKey OpenAI API key
 * @return array Response data
 */
function makeOpenAIRequest($data, $apiKey) {
    $options = [
        'http' => [
            'header' => [
                "Content-Type: application/json",
                "Authorization: Bearer " . $apiKey,
                "User-Agent: FashionChatbot/1.0"
            ],
            'method' => 'POST',
            'content' => json_encode($data),
            'timeout' => 90,
            'ignore_errors' => true
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents('https://api.openai.com/v1/chat/completions', false, $context);
    
    if ($result === false) {
        return [
            'success' => false,
            'error' => 'Failed to connect to OpenAI API. Please check your internet connection.',
            'answer' => 'I\'m having trouble connecting to the AI service right now. Please try again in a few moments.'
        ];
    }
    
    // Parse response
    $response = json_decode($result, true);
    
    if ($response === null) {
        return [
            'success' => false,
            'error' => 'Invalid JSON response from OpenAI',
            'answer' => 'I received an invalid response from the AI service. Please try again.'
        ];
    }
    
    // Handle API errors
    if (isset($response['error'])) {
        $errorMessage = $response['error']['message'] ?? 'Unknown API error';
        return [
            'success' => false,
            'error' => "OpenAI API Error: $errorMessage",
            'answer' => 'I encountered an error while processing your request. Please try again or rephrase your question.'
        ];
    }
    
    // Extract successful response
    if (isset($response['choices'][0]['message']['content'])) {
        return [
            'success' => true,
            'answer' => trim($response['choices'][0]['message']['content']),
            'model_used' => $data['model'],
            'timestamp' => date('Y-m-d H:i:s'),
            'tokens_used' => $response['usage']['total_tokens'] ?? 'unknown'
        ];
    }
    
    return [
        'success' => false,
        'error' => 'Unexpected response format from OpenAI',
        'answer' => 'I couldn\'t process your request properly. Please try again.'
    ];
}

/**
 * Log API requests for debugging and monitoring
 * @param array $data Data to log
 */
function logRequest($data) {
    try {
        $logFile = 'fashion_api_logs.txt';
        $timestamp = date('Y-m-d H:i:s');
        $logEntry = "[$timestamp] " . json_encode($data, JSON_UNESCAPED_UNICODE | JSON_PARTIAL_OUTPUT_ON_ERROR) . "\n";
        
        // Append to log file with file locking
        file_put_contents($logFile, $logEntry, FILE_APPEND | LOCK_EX);
    } catch (Exception $e) {
        error_log("Failed to write to log file: " . $e->getMessage());
    }
}

// Load environment variables
loadEnvironmentVariables(__DIR__ . '/.env');

// Handle CORS preflight requests
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Handle GET requests (health check)
if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    echo json_encode([
        'status' => 'healthy',
        'message' => 'Fashion Chatbot with Image Analysis is Connected',
        'version' => '2.0.0',
        'features' => [
            'text_chat',
            'image_analysis',
            'fashion_advice',
            'style_consultation'
        ],
        'timestamp' => date('Y-m-d H:i:s'),
        'server_time' => date('c')
    ]);
    exit();
}

// Handle POST requests (main chat functionality)
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    try {
        // Get request data
        $userId = $_POST['user_id'] ?? 'anonymous_' . uniqid();
        $message = $_POST['message'] ?? '';
        $hasImage = isset($_FILES['image']);
        
        // Log the request (excluding sensitive data)
        logRequest([
            'user_id' => $userId,
            'has_message' => !empty($message),
            'has_image' => $hasImage,
            'message_length' => strlen($message),
            'ip' => $_SERVER['REMOTE_ADDR'] ?? 'unknown',
            'user_agent' => substr($_SERVER['HTTP_USER_AGENT'] ?? 'unknown', 0, 100)
        ]);
        
        // Validate input
        if (empty(trim($message)) && !$hasImage) {
            echo json_encode([
                'success' => false,
                'error' => 'Please provide either a message or upload an image',
                'answer' => 'I need either a question or an image to help you with fashion advice. What can I assist you with today?'
            ]);
            exit();
        }
        
        // Get OpenAI API key
        $openaiKey = $_ENV['OPENAI_API_KEY'] ?? '';
        if (empty($openaiKey) || strpos($openaiKey, 'your_actual') !== false) {
            echo json_encode([
                'success' => false,
                'error' => 'OpenAI API key not properly configured',
                'answer' => '🔧 The AI service is not properly configured. Please contact the administrator to set up the OpenAI API key.'
            ]);
            exit();
        }
        
        // Process request based on whether image is included
        if ($hasImage) {
            // Handle image analysis
            $validation = validateImage($_FILES['image']);
            
            if (!$validation['valid']) {
                echo json_encode([
                    'success' => false,
                    'error' => $validation['error'],
                    'answer' => $validation['error'] . ' Please try uploading a different image.'
                ]);
                exit();
            }
            
            // Process image for API
            $imageBase64 = processImageForAPI($_FILES['image']['tmp_name'], $validation['type']);
            
            // Analyze image with OpenAI Vision
            $response = analyzeFashionImage($message, $imageBase64, $validation['type'], $openaiKey);
            
            // Add image info to response
            if ($response['success']) {
                $response['image_analyzed'] = true;
                $response['image_info'] = [
                    'type' => $validation['type'],
                    'size' => round($validation['size'] / 1024, 2) . ' KB',
                    'dimensions' => $validation['width'] . 'x' . $validation['height']
                ];
            }
            
        } else {
            // Handle text-only request
            $response = getFashionAdvice($message, $openaiKey);
            $response['image_analyzed'] = false;
        }
        
        // Add user info to response
        $response['user_id'] = $userId;
        $response['request_id'] = uniqid('req_');
        
        echo json_encode($response);
        
    } catch (Exception $e) {
        // Log the error
        error_log("Fashion Chatbot API Error: " . $e->getMessage());
        
        echo json_encode([
            'success' => false,
            'error' => 'Internal server error: ' . $e->getMessage(),
            'answer' => '🚨 I encountered an unexpected error while processing your request. Please try again, and if the problem persists, contact support.',
            'error_id' => uniqid('err_'),
            'timestamp' => date('Y-m-d H:i:s')
        ]);
    }
    exit();
}

// Handle unsupported methods
http_response_code(405);
echo json_encode([
    'success' => false,
    'error' => 'Method not allowed',
    'answer' => 'This endpoint only supports GET and POST requests.'
]);
?>
