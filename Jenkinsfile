pipeline {
  agent any

  triggers {
    githubPush()
  }

  options {
    timestamps()
    ansiColor('xterm')
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
          python3 -m venv ${VENV_DIR}
          . ${VENV_DIR}/bin/activate
          pip install --upgrade pip
          pip install -e .[dev]
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
      emailext(
        to: "${ALERT_EMAIL_TO}",
        subject: "[SUCCESS] ${JOB_NAME} #${BUILD_NUMBER}",
        body: "Build succeeded: ${BUILD_URL}"
      )
    }

    failure {
      sh '''
        . ${VENV_DIR}/bin/activate
        python scripts/notify.py --status failure --message "Metaflow pipeline failed"
      '''
      emailext(
        to: "${ALERT_EMAIL_TO}",
        subject: "[FAILURE] ${JOB_NAME} #${BUILD_NUMBER}",
        body: "Build failed: ${BUILD_URL}"
      )
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
