<?php
declare(strict_types=1);

namespace OCA\FaceRecognition\Helper;

use OCP\Image;
use OCP\ITempManager;
use OCA\FaceRecognition\Helper\Imaginary;

class TempImage extends Image {

        /** @var Imaginary */
        private $imaginary;

        /** @var string */
        private $imagePath;

        /** @var string */
        private $tempPath;

        /** @var string */
        private $preferredMimeType;

        /** @var int */
        private $maxImageArea;

        /** @var ITempManager */
        private $tempManager;

        /** @var int */
        private $minImageSide;

        /** @var float */
        private $ratio = -1.0;

        /** @var bool */
        private $skipped = false;

        public function __construct(string $imagePath,
                                    string $preferredMimeType,
                                    int    $maxImageArea,
                                    int    $minImageSide)
        {
                parent::__construct();

                $this->imagePath         = $imagePath;
                $this->preferredMimeType = $preferredMimeType;
                $this->maxImageArea      = $maxImageArea;
                $this->minImageSide      = $minImageSide;

                $this->tempManager       = \OC::$server->getTempManager();
                $this->imaginary         = new Imaginary();

                $this->prepareImage();
        }

        public function getTempPath(): string {
                return $this->tempPath;
        }

        public function getRatio(): float {
                return $this->ratio;
        }

        public function getSkipped(): bool {
                return $this->skipped;
        }

        public function clean() {
                $this->tempManager->clean();
        }

        private function prepareImage() {

                if ($this->imaginary->isEnabled()) {
                        // (Kod obsługi Imaginary bez zmian...)
                        $fileInfo = $this->imaginary->getInfo($this->imagePath);
                        $widthOrig = $fileInfo['width'];
                        $heightOrig = $fileInfo['height'];
                        if (($widthOrig < $this->minImageSide) || ($heightOrig < $this->minImageSide)) {
                                $this->skipped = true;
                                return;
                        }
                        $scaleFactor = $this->getResizeRatio($widthOrig, $heightOrig);
                        $newWidth = intval(round($widthOrig * $scaleFactor));
                        $newHeight = intval(round($heightOrig * $scaleFactor));
                        $resizedResource = $this->imaginary->getResized($this->imagePath, $newWidth, $newHeight, $fileInfo['autorotate'], $this->preferredMimeType);
                        $this->loadFromData($resizedResource);
                        if (!$this->valid()) {
                                throw new \RuntimeException("Imaginary image response is not valid.");
                        }
                        $this->ratio = 1 / $scaleFactor;
                }
                else {
                        // --- FIX HEIC (SAFE TEMP VERSION) ---
                        $ext = strtolower(pathinfo($this->imagePath, PATHINFO_EXTENSION));
                        if ($ext === 'heic' || $ext === 'heif') {
                            // 1. Tworzymy bezpieczną ścieżkę w folderze /tmp (nie w albumie użytkownika!)
                            // Nextcloud wygeneruje np: /tmp/oc_tmp_A7b29.jpg
                            $safeTempJpg = $this->tempManager->getTemporaryFile('.jpg');
                            
                            // 2. Konwersja: Źródło -> Bezpieczny TEMP
                            // Używamy 'convert' systemowego
                            $cmd = "convert " . escapeshellarg($this->imagePath) . " " . escapeshellarg($safeTempJpg) . " 2>&1";
                            exec($cmd, $output, $returnCode);

                            if ($returnCode === 0 && file_exists($safeTempJpg)) {
                                // 3. Sukces: Podmieniamy ścieżkę źródłową na nasz tymczasowy JPG
                                // Od teraz Nextcloud myśli, że pracuje na pliku z /tmp, a nie na oryginale z iPhone'a
                                $this->imagePath = $safeTempJpg;
                            } else {
                                // Błąd: Rzucamy wyjątek z logiem
                                $errorMsg = "HEIC Conversion Failed. CMD: $cmd. Output: " . implode(" | ", $output);
                                throw new \RuntimeException($errorMsg);
                            }
                        }
                        // --- END FIX ---

                        $this->loadFromFile($this->imagePath);
                        $this->fixOrientation();

                        if (!$this->valid()) {
                                $errorMsg = "Local image is not valid. Path: " . $this->imagePath;
                                if (function_exists('ini_get')) {
                                    $errorMsg .= " (PHP RAM: " . ini_get('memory_limit') . ")";
                                }
                                throw new \RuntimeException($errorMsg);
                        }

                        if ((imagesx($this->resource()) < $this->minImageSide) ||
                            (imagesy($this->resource()) < $this->minImageSide)) {
                                $this->skipped = true;
                                return;
                        }

                        $this->ratio = $this->resizeOCImage();
                }

                $this->tempPath = $this->tempManager->getTemporaryFile();
                $this->save($this->tempPath, $this->preferredMimeType);
        }

        private function resizeOCImage(): float {
                $widthOrig = imagesx($this->resource());
                $heightOrig = imagesy($this->resource());

                if (($widthOrig <= 0) || ($heightOrig <= 0)) {
                        $message = "Image is having non-positive width or height, cannot continue";
                        throw new \RuntimeException($message);
                }

                $scaleFactor = $this->getResizeRatio($widthOrig, $heightOrig);

                $newWidth = intval(round($widthOrig * $scaleFactor));
                $newHeight = intval(round($heightOrig * $scaleFactor));

                $success = $this->preciseResize($newWidth, $newHeight);
                if ($success === false) {
                        throw new \RuntimeException("Error during image resize");
                }

                return 1 / $scaleFactor;
        }

        private function getResizeRatio($widthOrig, $heightOrig): float {
                $areaRatio = $this->maxImageArea / ($widthOrig * $heightOrig);
                return sqrt($areaRatio);
        }
}
