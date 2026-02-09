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
          if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
            echo "Configured PYTHON_BIN version is too old."
            "${PYTHON_BIN}" --version || true
            echo "Please set PYTHON_BIN to Python 3.10+."
            exit 1
          fi
          "${PYTHON_BIN}" -m venv --clear ${VENV_DIR}
          ${VENV_DIR}/bin/python3 --version
          ${VENV_DIR}/bin/python3 -m pip --version
          ${VENV_DIR}/bin/python3 -m pip install --upgrade pip
          ${VENV_DIR}/bin/python3 -m pip install -e '.[dev]'
        '''
      }
    }

    stage('Unit Tests') {
      steps {
        sh '''
          ${VENV_DIR}/bin/python3 -m pytest -q
        '''
      }
    }

    stage('Run Metaflow') {
      steps {
        sh '''
          if [ "${USE_STEP_FUNCTIONS}" = "true" ]; then
            ${VENV_DIR}/bin/python3 flows/fraud_detection_flow.py --with step-functions --with batch create || true
            ${VENV_DIR}/bin/python3 flows/fraud_detection_flow.py --with step-functions trigger
          else
            ${VENV_DIR}/bin/python3 flows/fraud_detection_flow.py run --sample-size 25000
          fi
        '''
      }
    }

    stage('Log Metaflow Run') {
      steps {
        sh '''
          ${VENV_DIR}/bin/python3 scripts/log_metaflow_run.py
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
        ${VENV_DIR}/bin/python3 scripts/notify.py --status success --message "Metaflow pipeline completed"
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
        ${VENV_DIR}/bin/python3 scripts/notify.py --status failure --message "Metaflow pipeline failed"
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
      script {
        try {
          slackSend(
            channel: '#mlops-alerts',
            color: 'danger',
            message: "Build failed: ${JOB_NAME} #${BUILD_NUMBER} ${BUILD_URL}"
          )
        } catch (Throwable err) {
          echo "slackSend unavailable; skipping Slack notification. ${err.class.simpleName}"
        }
      }
    }

    always {
      archiveArtifacts artifacts: 'artifacts/**,data/lakehouse/**', allowEmptyArchive: true, fingerprint: true
    }
  }
}
