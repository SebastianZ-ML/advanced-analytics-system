import { ChatAnswer, DashboardSpec, DataCatalog, DataQualityReport, InsightReport, ObjectiveSpec, Project, RelationshipSpec, RunEvent, TransformationRecord, ValidationReport } from './types';

const API_BASE = '/api';

export const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },

  async listProjects(): Promise<Project[]> {
    const res = await fetch(`${API_BASE}/projects`);
    return res.json();
  },

  async createProject(name: string, description: string = ''): Promise<Project> {
    const res = await fetch(`${API_BASE}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    return res.json();
  },

  async getProject(projectId: string): Promise<Project> {
    const res = await fetch(`${API_BASE}/projects/${projectId}`);
    return res.json();
  },

  async loadDemoDataset(projectId: string) {
    const res = await fetch(`${API_BASE}/projects/${projectId}/load_demo`, {
      method: 'POST',
    });
    return res.json();
  },

  async uploadFile(projectId: string, file: File, sheetName?: string, tableName?: string) {
    const formData = new FormData();
    formData.append('file', file);
    if (sheetName) formData.append('sheet_name', sheetName);
    if (tableName) formData.append('table_name', tableName);

    const res = await fetch(`${API_BASE}/projects/${projectId}/upload`, {
      method: 'POST',
      body: formData,
    });
    return res.json();
  },

  async startRun(
    projectId: string,
    objectiveQuestion?: string,
    userClarifications?: Record<string, string>,
    forceErrorForTest: boolean = false
  ) {
    const res = await fetch(`${API_BASE}/projects/${projectId}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        objective_question: objectiveQuestion,
        user_clarifications: userClarifications,
        force_error_for_test: forceErrorForTest,
      }),
    });
    return res.json();
  },

  async getRunStatus(runId: string) {
    const res = await fetch(`${API_BASE}/runs/${runId}`);
    return res.json();
  },

  async getRunArtifacts(runId: string): Promise<{
    ObjectiveSpec?: ObjectiveSpec;
    DataCatalog?: DataCatalog;
    DataQualityReport?: DataQualityReport;
    Relationships?: RelationshipSpec[];
    AnalysisPlan?: any;
    TransformationRecords?: TransformationRecord[];
    AnalysisResults?: any[];
    ValidationReport?: ValidationReport;
    InsightReport?: InsightReport;
    DashboardSpec?: DashboardSpec;
  }> {
    const res = await fetch(`${API_BASE}/runs/${runId}/artifacts`);
    return res.json();
  },

  async getRunEvents(runId: string): Promise<RunEvent[]> {
    const res = await fetch(`${API_BASE}/runs/${runId}/events`);
    return res.json();
  },

  async chat(projectId: string, runId: string, question: string, activeFilters: Record<string, any> = {}): Promise<ChatAnswer> {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: projectId,
        run_id: runId,
        question,
        active_filters: activeFilters,
      }),
    });
    return res.json();
  },
};
