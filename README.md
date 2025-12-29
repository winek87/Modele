Trzeba updatowac facerecognition

sudo -u www-data cp TempImage.php /var/www/nextcloud/apps/facerecognition/lib/Helper/TempImage.php

1. Bezpieczeństwo danych: Używamy $this->tempManager->getTemporaryFile('.jpg'). Nextcloud sam wybierze bezpieczny folder tymczasowy (zazwyczaj /tmp/ lub /var/www/nextcloud/data/tmp/).

2. Brak konfliktów: Wygenerowana nazwa będzie losowa (np. oc_tmp_k8s2a.jpg). Nawet jeśli masz plik 1.jpg w albumie, ten proces go w ogóle nie dotknie.

3. Czystość: Nextcloud automatycznie sprząta pliki tymczasowe utworzone przez tempManager, więc nie zostawisz śmieci na serwerze.
