pipeline {
  agent any

  triggers {
    githubPush()
  }

  options {
    timestamps()
  }

  parameters {
    booleanParam(name: 'USE_STEP_FUNCTIONS', defaultValue: false, description: 'Run on AWS Step Functions + Batch')
    choice(name: 'DEPLOY_TARGET', choices: ['fastapi', 'sagemaker', 'k8s'], description: 'Model deployment target')
  }

  environment {
    VENV_DIR = '.venv'
    PYTHONUNBUFFERED = '1'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Install Dependencies') {
      steps {
        sh '''
          PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3}"
          echo "Using python binary: ${PYTHON_BIN}"
          if [ ! -x "${PYTHON_BIN}" ]; then
            echo "Configured PYTHON_BIN is not executable: ${PYTHON_BIN}"
            echo "Set PYTHON_BIN in Jenkins to your Python 3.10+ path."
            exit 1
          fi
          "${PYTHON_BIN}" -m venv ${VENV_DIR}
          . ${VENV_DIR}/bin/activate
          python --version
          pip --version
          pip install --upgrade pip
          pip install -e '.[dev]'
        '''
      }
    }

    stage('Unit Tests') {
      steps {
        sh '''
          . ${VENV_DIR}/bin/activate
          pytest -q
        '''
      }
    }

    stage('Run Metaflow') {
      steps {
        sh '''
          . ${VENV_DIR}/bin/activate
          if [ "${USE_STEP_FUNCTIONS}" = "true" ]; then
            python flows/fraud_detection_flow.py --with step-functions --with batch create || true
            python flows/fraud_detection_flow.py --with step-functions trigger
          else
            python flows/fraud_detection_flow.py run --sample-size 25000
          fi
        '''
      }
    }

    stage('Log Metaflow Run') {
      steps {
        sh '''
          . ${VENV_DIR}/bin/activate
          python scripts/log_metaflow_run.py
        '''
      }
    }

    stage('Deploy Model') {
      when {
        allOf {
          branch 'main'
          expression { return env.MODEL_S3_URI?.trim() }
        }
      }
      steps {
        sh '''
          . ${VENV_DIR}/bin/activate
          bash scripts/deploy_from_s3.sh "${MODEL_S3_URI}" "${DEPLOY_TARGET}"
        '''
      }
    }
  }

  post {
    success {
      sh '''
        . ${VENV_DIR}/bin/activate
        python scripts/notify.py --status success --message "Metaflow pipeline completed"
      '''
      script {
        def alertEmailToSafe = env.ALERT_EMAIL_TO ?: ""
        if (alertEmailToSafe.trim()) {
          emailext(
            to: alertEmailToSafe,
            subject: "[SUCCESS] ${JOB_NAME} #${BUILD_NUMBER}",
            body: "Build succeeded: ${BUILD_URL}"
          )
        } else {
          echo 'ALERT_EMAIL_TO is empty; skipping success email notification.'
        }
      }
    }

    failure {
      sh '''
        . ${VENV_DIR}/bin/activate
        python scripts/notify.py --status failure --message "Metaflow pipeline failed"
      '''
      script {
        def alertEmailToSafe = env.ALERT_EMAIL_TO ?: ""
        if (alertEmailToSafe.trim()) {
          emailext(
            to: alertEmailToSafe,
            subject: "[FAILURE] ${JOB_NAME} #${BUILD_NUMBER}",
            body: "Build failed: ${BUILD_URL}"
          )
        } else {
          echo 'ALERT_EMAIL_TO is empty; skipping failure email notification.'
        }
      }
      slackSend(
        channel: '#mlops-alerts',
        color: 'danger',
        message: "Build failed: ${JOB_NAME} #${BUILD_NUMBER} ${BUILD_URL}"
      )
    }

    always {
      archiveArtifacts artifacts: 'artifacts/**,data/lakehouse/**', allowEmptyArchive: true, fingerprint: true
    }
  }
}
